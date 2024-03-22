"""
File: data_loader.py
Project: dataloader
Created Date: 2023-10-19 02:24:47
Author: chenkaixu
-----
Comment:
A pytorch lightning data module based dataloader, for train/val/test dataset prepare.
Have a good code time!
-----
Last Modified: 2023-10-29 12:18:59
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

22-03-2024	Kaixu Chen	add different class num mapping dict. In collate_fn, re-mapping the label from .json disease key.

"""

# %%

import logging, json, shutil, os
from pathlib import Path

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Resize,
    RandomHorizontalFlip,
    ToTensor,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
    Div255,
    create_video_transform,
)

from typing import Any, Callable, Dict, Optional, Type
from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset

from gait_video_dataset import labeled_gait_video_dataset

disease_to_num_mapping_Dict: Dict = {
    2: {
        "ASD": 0,
        "non-ASD": 1
    },
    3: {
        "ASD": 0,
        "DHS": 1,
        "LCS_HipOA": 2
    },
    4: {
        "ASD": 0,
        "DHS": 1,
        "LCS_HipOA": 2,
        "normal": 3
    }
}

class WalkDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict = None):
        super().__init__()

        self._seg_path = opt.data.seg_data_path
        self._gait_seg_path = opt.data.gait_seg_data_path
        self.gait_cycle = opt.train.gait_cycle

        self._BATCH_SIZE = opt.train.batch_size
        self._NUM_WORKERS = opt.data.num_workers
        self._IMG_SIZE = opt.data.img_size

        # frame rate
        self._CLIP_DURATION = opt.train.clip_duration
        self.uniform_temporal_subsample_num = opt.train.uniform_temporal_subsample_num

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx
        self._class_num = opt.model.model_class_num

        self._temporal_mix = opt.train.temporal_mix

        self.train_mapping_transform = Compose(
            [
                UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                Div255(),
                Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
            ]
        )

        self.train_video_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video", 
                    transform=Compose(
                        [
                            Div255(),
                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                        ])
                ),
            ]
        )

        self.val_video_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Div255(),
                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            UniformTemporalSubsample(
                                self.uniform_temporal_subsample_num
                            ),
                        ]
                    ),
                ),
            ]
        )

    def prepare_data(self) -> None:
        """here prepare the temp val data path,
        because the val dataset not use the gait cycle index,
        so we directly use the pytorchvideo API to load the video.
        AKA, use whole video to validate the model.
        """

        # temp_val_path = Path(self._seg_path) / "val_temp"
        # val_idx = self._dataset_idx[1]

        # shutil.rmtree(temp_val_path, ignore_errors=True)

        # for path in val_idx:
        #     with open(path) as f:
        #         file_info_dict = json.load(f)

        #     video_name = file_info_dict["video_name"]
        #     video_path = file_info_dict["video_path"]
        #     video_disease = file_info_dict["disease"]

        #     if not (temp_val_path / video_disease).exists():
        #         (temp_val_path / video_disease).mkdir(parents=True, exist_ok=False)

        #     shutil.copy(
        #         video_path, temp_val_path / video_disease / (video_name + ".mp4")
        #     )

        # self.temp_val_path = temp_val_path

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        if stage in ("fit", None):
            # judge if use split the gait cycle, or use the whole video.
            if self.gait_cycle == -1:
                self.train_gait_dataset = labeled_video_dataset(
                    data_path=self._dataset_idx[1],
                    clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                    transform=self.train_video_transform,
                )

            else:
                # * labeled dataset, where first define the giat cycle, and from .json file to load the video.
                # * here only need dataset idx, mean first split the dataset, and then load the video.
                self.train_gait_dataset = labeled_gait_video_dataset(
                    gait_cycle = self.gait_cycle,
                    dataset_idx=self._dataset_idx[0],  # [train, val]
                    transform=self.train_mapping_transform,
                    temporal_mix=self._temporal_mix
                )

        if stage in ("fit", "validate", None):
            # * the val dataset, do not apply the gait cycle, just load the whole video.
            self.val_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx[2],
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=self.val_video_transform,
            )

        if stage in ("test", None):
            # * the test dataset, do not apply the gait cycle, just load the whole video.
            self.test_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx[2],
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=self.val_video_transform,
            )

    def collate_fn(self, batch):
        """this function process the batch data, and return the batch data.

        Args:
            batch (list): the batch from the dataset.
            The batch include the one patient info from the json file.
            Here we only cat the one patient video tensor, and label tensor.

        Returns:
            dict: {video: torch.tensor, label: torch.tensor, info: list}
        """

        batch_label = []
        batch_video = []

        # mapping label 

        for i in batch:
            # logging.info(i['video'].shape)
            gait_num, c, t, h, w = i["video"].shape
            disease = i["disease"]

            batch_video.append(i["video"])
            for _ in range(gait_num):
                
                if disease in disease_to_num_mapping_Dict[self._class_num].keys():

                    batch_label.append(disease_to_num_mapping_Dict[self._class_num][disease])
                else:
                    # * if the disease not in the mapping dict, then set the label to non-ASD.
                    batch_label.append(disease_to_num_mapping_Dict[self._class_num]['non-ASD'])

        # video, b, c, t, h, w, which include the video frame from sample info
        # label, b, which include the video frame from sample info
        # sample info, the raw sample info from dataset
        return {
            "video": torch.cat(batch_video, dim=0),
            "label": torch.tensor(batch_label),
            "info": batch,
        }

    def train_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
        if self.gait_cycle == -1:
            train_data_loader = DataLoader(
                self.train_gait_dataset,
                batch_size=self._BATCH_SIZE,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
            )

        else:
            train_data_loader = DataLoader(
            self.train_gait_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        return DataLoader(
            self.val_gait_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
            
        return DataLoader(
            self.test_gait_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )
