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


class WalkDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict = None):
        super().__init__()

        self._DATA_PATH = opt.data.data_path
        self._TARIN_PATH = opt.train.train_path

        self._seg_path = opt.data.seg_data_path
        self._gait_seg_path = opt.data.gait_seg_data_path

        self._BATCH_SIZE = opt.train.batch_size
        self._NUM_WORKERS = opt.data.num_workers
        self._IMG_SIZE = opt.data.img_size

        # frame rate
        self._CLIP_DURATION = opt.train.clip_duration
        self.uniform_temporal_subsample_num = opt.train.uniform_temporal_subsample_num

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx

        self.train_transform = Compose(
            [
                UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                Div255(),
                Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
            ]
        )

        self.val_transform = Compose(
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

        temp_val_path = Path(self._seg_path) / "val_temp"
        val_idx = self._dataset_idx[1]

        shutil.rmtree(temp_val_path, ignore_errors=True)

        for path in val_idx:
            with open(path) as f:
                file_info_dict = json.load(f)

            video_name = file_info_dict["video_name"]
            video_path = file_info_dict["video_path"]
            video_disease = file_info_dict["disease"]

            if not (temp_val_path / video_disease).exists():
                (temp_val_path / video_disease).mkdir(parents=True, exist_ok=False)

            shutil.copy(
                video_path, temp_val_path / video_disease / (video_name + ".mp4")
            )

        self.temp_val_path = temp_val_path

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        print("#" * 100)
        print("run pre process model!", self._TARIN_PATH)
        print("#" * 100)

        if stage in ("fit", None):
            # * labeled dataset, where first define the giat cycle, and from .json file to load the video.
            # * here only need dataset idx, mean first split the dataset, and then load the video.
            self.train_gait_dataset = labeled_gait_video_dataset(
                dataset_idx=self._dataset_idx[0],  # [train, val]
                transform=self.train_transform,
            )

        if stage in ("fit", "validate", None):
            # * the val dataset, do not apply the gait cycle, just load the whole video.
            self.val_gait_dataset = labeled_video_dataset(
                data_path=self.temp_val_path,
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=self.val_transform,
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

        for i in batch:
            # logging.info(i['video'].shape)
            gait_num, c, t, h, w = i["video"].shape

            batch_video.append(i["video"])
            for _ in range(gait_num):
                batch_label.append(i["label"])

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
        return DataLoader(
            self.train_gait_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

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
