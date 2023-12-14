#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/dataloader/gait_video_dataset.py
Project: /workspace/skeleton/project/dataloader
Created Date: Monday December 11th 2023
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday December 11th 2023 11:26:34 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2023 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from __future__ import annotations

import logging, sys, json

sys.path.append("/workspace/skeleton")

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch
import torch.utils.data

from torchvision.io import read_video

from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from pytorchvideo.data.video import Video, VideoPathHandler
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.utils import MultiProcessSampler

logger = logging.getLogger(__name__)

class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        transform: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        super().__init__()

        self._transform = transform
        self._labeled_videos = self.convert_dict_to_list(labeled_video_paths)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

    @staticmethod
    def convert_dict_to_list(labeled_video_path):
        ans_list = []

        # in load case, we do not need to split the disease.
        for disease, video_path in labeled_video_path.items():
            for i in video_path:
                ans_list.append(i)

        return ans_list
    
    @staticmethod
    def split_gait_cycle(video_tensor: torch.Tensor, gait_cycle_index: list):
        # TODO: 这个方法需要可以区别不同的步行周期
        # 也就是说需要根据给定的参数，能够区分不同的步行周期
        # 例如， 2分的话，需要能区分前后， 4分的话，需要能区分前后，中间，等

        ans_list = []
        for i in range(0, len(gait_cycle_index)-1, 2):
            ans_list.append(video_tensor[gait_cycle_index[i]:gait_cycle_index[i+1], ...])

        return ans_list # needed gait cycle video tensor

    def __len__(self):
        return len(self._labeled_videos)

    def __getitem__(self, index) -> Any:

        # load the video tensor from json file
        with open(self._labeled_videos[index]) as f:
            file_info_dict = json.load(f)

        # load video info from json file
        video_name = file_info_dict["video_name"]
        video_path = file_info_dict["video_path"]
        vframes, _, _ = read_video(video_path, output_format="TCHW")
        label = file_info_dict["label"]
        disease = file_info_dict["disease"]
        gait_cycle_index_bbox = file_info_dict["gait_cycle_index_bbox"]
        bbox_none_index = file_info_dict["bbox_none_index"]

        logging.info(f"video name: {video_name}, gait cycle index: {gait_cycle_index_bbox}")

        defined_vframes = self.split_gait_cycle(vframes, gait_cycle_index_bbox)

        sample_info_dict = {
            "video": defined_vframes,
            "label": label,
            "disease": disease,
            "video_name": video_name,
            "video_index": index,
            "gait_index": gait_cycle_index_bbox,
            "bbox_none_index": bbox_none_index,
        }

        if self._transform is not None:
            video_t_list = []
            for video_t in defined_vframes:
                video_t_list.append(self._transform(video_t.permute(1, 0, 2, 3)))

            sample_info_dict["video"] = torch.stack(video_t_list, dim=0) # c, t, h, w
        else:
            print("no transform")

        return sample_info_dict


def labeled_gait_video_dataset(
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset_idx: Dict = None,
) -> LabeledGaitVideoDataset:

    dataset = LabeledGaitVideoDataset(
        dataset_idx,
        transform,
    )

    return dataset
