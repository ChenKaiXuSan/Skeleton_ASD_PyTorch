'''
File: data_loader.py
Project: dataloader
Created Date: 2023-10-19 02:24:47
Author: chenkaixu
-----
Comment:
A pytorch lightning data module based dataloader, for train/val/test dataset prepare.
Have a good code time!
-----
Last Modified: 2023-10-19 02:36:30
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

'''

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

from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset, labeled_video_dataset

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
    '''
    A helper function to create "LabeledVideoDataset" object for the Walk dataset.

    Args:
        data_path (str): Path to the data. The path defines how the data should be read. For a directory, the directory structure defines the classes (i.e. each subdirectory is class).
        clip_sampler (ClipSampler): Defines how clips should be sampled from each video. See the clip sampling documentation for more information.
        video_sampler (Type[torch.utils.data.Sampler], optional): Sampler for the internal video container. Defaults to torch.utils.data.RandomSampler.
        transform (Optional[Callable[[Dict[str, Any]], Dict[str, Any]]], optional): This callable is evaluated on the clip output before the clip is returned. Defaults to None.
        video_path_prefix (str, optional): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. Defaults to "".
        decode_audio (bool, optional): If True, also decode audio from video. Defaults to False. Notice that, if Ture will trigger the stack error.
        decoder (str, optional): Defines what type of decoder used to decode a video. Defaults to "pyav".

    Returns:
        LabeledVideoDataset: _description_
    '''
    return labeled_video_dataset(
        data_path,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder
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

        # frame rate
        self._CLIP_DURATION = opt.train.clip_duration
        self.uniform_temporal_subsample_num = opt.train.uniform_temporal_subsample_num

        self.train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            # uniform clip T frames from the given n sec video.
                            UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                            
                            # dived the pixel from [0, 255] tp [0, 1], to save computing resources.
                            Div255(),
                            # Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),

                            # RandomShortSideScale(min_size=256, max_size=320),
                            # RandomCrop(self._IMG_SIZE),

                            # ShortSideScale(self._IMG_SIZE),

                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
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
                    )
                ),
            ]
        )

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        '''
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        '''

        print("#" * 100)
        print("run pre process model!", self._TARIN_PATH)
        print("#" * 100)

        # if stage == "fit" or stage == None:
        if stage in ("fit", None):
            self.train_dataset = WalkDataset(
                data_path=os.path.join(self._TARIN_PATH, "train"),
                clip_sampler=make_clip_sampler("random", self._CLIP_DURATION),
                transform=self.train_transform,
            )

        if stage in ("fit", "validate", None):
            self.val_dataset = WalkDataset(
                data_path=os.path.join(self._TARIN_PATH, "val"),
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=self.train_transform,
            )

        if stage in ("predict", "test", None):
            self.test_pred_dataset = WalkDataset(
                data_path=os.path.join(self._TARIN_PATH, "val"),
                clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=self.train_transform
            )

    def train_dataloader(self) -> DataLoader:
        '''
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        '''
        return DataLoader(
            self.train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        '''
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        '''

        return DataLoader(
            self.val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        '''
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and 
        normalizes the video before applying the scale, crop and flip augmentations.
        '''
        return DataLoader(
            self.test_pred_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )
    