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

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Resize,
    RandomHorizontalFlip,
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
import os

import torch
from torch.utils.data import DataLoader
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data import make_clip_sampler

from pytorchvideo.data.labeled_video_dataset import (
    LabeledVideoDataset,
    labeled_video_dataset,
)

from labeled_video_dataset import LabeledGaitVideoDataset, labeled_gait_video_dataset

# %%

def WalkDataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = False,
    decoder: str = "pyav",
) -> LabeledVideoDataset:

    return labeled_video_dataset(
        data_path,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder,
    )

def WalkGaitDataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = False,
    decoder: str = "pyav",
    yolo_config: Dict = None,
) -> LabeledVideoDataset:

    return labeled_gait_video_dataset(
        data_path,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder,
        yolo_config
    )

# %%


class WalkDataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self._DATA_PATH = opt.data.data_path
        self._TARIN_PATH = opt.train.train_path

        self._BATCH_SIZE = opt.train.batch_size
        self._NUM_WORKERS = opt.data.num_workers
        self._IMG_SIZE = opt.data.img_size

        self._yolo_config = opt

        # frame rate
        self._CLIP_DURATION = opt.train.clip_duration
        self.uniform_temporal_subsample_num = opt.train.uniform_temporal_subsample_num

        # self.train_transform = Compose(
        #     [
        #         ApplyTransformToKey(
        #             key="video",
        #             transform=Compose(
        #                 [
        #                     # uniform clip T frames from the given n sec video.
        #                     UniformTemporalSubsample(
        #                         self.uniform_temporal_subsample_num
        #                     ),
        #                     # dived the pixel from [0, 255] tp [0, 1], to save computing resources.
        #                     Div255(),
        #                     # Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        #                     # RandomShortSideScale(min_size=256, max_size=320),
        #                     # RandomCrop(self._IMG_SIZE),
        #                     # ShortSideScale(self._IMG_SIZE),
        #                     Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
        #                     RandomHorizontalFlip(p=0.5),
        #                 ]
        #             ),
        #         ),
        #     ]
        # )

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
                        ]
                    ),
                ),
            ]
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        print("#" * 100)
        print("run pre process model!", self._TARIN_PATH)
        print("#" * 100)

        # if stage == "fit" or stage == None:
        if stage in ("fit", None):
            # self.train_dataset = WalkDataset(
            #     data_path=os.path.join(self._TARIN_PATH, "train"),
            #     clip_sampler=make_clip_sampler("random", self._CLIP_DURATION),
            #     transform=self.train_transform,
            # )

            self.train_gait_dataset = WalkGaitDataset(
                data_path=os.path.join(self._TARIN_PATH, "train"),
                clip_sampler=make_clip_sampler("random", self._CLIP_DURATION),
                transform=self.train_transform,
                yolo_config = self._yolo_config
            )

        if stage in ("fit", "validate", None):
            # self.val_dataset = WalkDataset(
            #     data_path=os.path.join(self._TARIN_PATH, "val"),
            #     clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
            #     transform=self.train_transform,
            # )

            self.val_gait_dataset = WalkDataset(
                data_path=os.path.join(self._TARIN_PATH, "val"),
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=self.val_transform,
            )

        # if stage in ("predict", "test", None):
        #     self.test_pred_dataset = WalkDataset(
        #         data_path=os.path.join(self._TARIN_PATH, "val"),
        #         clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
        #         transform=self.train_transform,
        #     )

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
            shuffle=False,
            drop_last=True,
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

    def test_dataloader(self) -> DataLoader:
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
