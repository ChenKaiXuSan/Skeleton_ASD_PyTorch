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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch

from torchvision.io import read_video, write_png

logger = logging.getLogger(__name__)


class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        gait_cycle: int,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        transform: Optional[Callable[[dict], Any]] = None,
        temporal_mix: bool = False,
    ) -> None:
        super().__init__()

        self._transform = transform
        self._labeled_videos = labeled_video_paths
        self._gait_cycle = gait_cycle
        self._temporal_mix = temporal_mix

    def temporal_mix(self, video_tensor: torch.Tensor, gait_cycle_index: list, bbox: torch.Tensor) -> list:

        first_phase, first_phase_idx = self.split_gait_cycle(video_tensor, gait_cycle_index, 0)
        second_phase, second_phase_idx = self.split_gait_cycle(video_tensor, gait_cycle_index, 1)

        # check if first phse and second phase have the same length
        if len(first_phase) > len(second_phase):
            second_phase.append(second_phase[-1])
            second_phase_idx.append(second_phase_idx[-1])
        elif len(first_phase) < len(second_phase):
            first_phase.append(first_phase[-1])
            first_phase_idx.append(first_phase_idx[-1])
            
        assert len(first_phase) == len(second_phase), "first phase and second phase have different length"

        # TODO: fuse the different phase to generate new frame.
        for i in range(len(first_phase)):
            first = first_phase[i]
            second = second_phase[i]
            
            first_idx = first_phase_idx[i]
            second_idx = second_phase_idx[i]

            cropped_first_frame = []
            cropped_second_frame = []

            # find bbox 
            for j in range(len(first)):
                first_one_frame_bbox = bbox[first_idx+j] # xywh

                x, y, w, h = first_one_frame_bbox
                xmin = int(x - w / 2)
                ymin = int(y - h / 2)
                xmax = int(x + w / 2)
                ymax = int(y + h / 2)

                # split human with bbox 
                cropped_first_one_frame_human = first[j][:, ymin:ymax, xmin:xmax]
                cropped_first_frame.append(cropped_first_one_frame_human)

                write_png(input=cropped_first_one_frame_human, filename='first_text.png')
            
            # TODO: here should find the max width to stack them
            cropped_first_frame = torch.stack(cropped_first_frame, dim=1) # c, t, h, w

            for j in range(len(second)):
                second_one_frame_bbox = bbox[second_idx+j] # xyxy

                x, y, w, h = second_one_frame_bbox
                xmin = int(x - w / 2)
                ymin = int(y - h / 2)
                xmax = int(x + w / 2)
                ymax = int(y + h / 2)

                # crop human with bbox 
                cropped_second_one_frame_human = second[j][:, ymin:ymax, xmin:xmax]
                cropped_second_frame.append(cropped_second_one_frame_human)

                write_png(input=cropped_second_one_frame_human, filename='second_text.png')
            # TODO: here should find the max width to pad and stack them 
            cropped_second_frame = torch.stack(cropped_second_frame, dim=1) # c, t, h, w
                
            # uniform the frame length
            if self._transform is not None:
                cropped_first_frame = self._transform(cropped_first_frame) # c, t, h, w
                cropped_second_frame = self._transform(cropped_second_frame) # c, t, h, w

            else:
                print("no transform")

            # fuse the frame
            new_frame = torch.cat([cropped_first_frame, cropped_second_frame], dim=0)

    @staticmethod
    def split_gait_cycle(video_tensor: torch.Tensor, gait_cycle_index: list, gait_cycle: int):
        # TODO: 这个方法需要可以区别不同的步行周期
        # 也就是说需要根据给定的参数，能够区分不同的步行周期
        # 例如， 2分的话，需要能区分前后， 4分的话，需要能区分前后，中间，等

        use_idx = []
        ans_list = []
        if gait_cycle == 0:
            for i in range(0, len(gait_cycle_index)-1, 2):
                ans_list.append(video_tensor[gait_cycle_index[i]:gait_cycle_index[i+1], ...])
                use_idx.append(gait_cycle_index[i])
        elif gait_cycle == 1:
            if len(gait_cycle_index) == 2:
                ans_list.append(video_tensor[gait_cycle_index[0]:gait_cycle_index[1], ...])
                use_idx.append('the gait cycle index is less than 2, so use first gait cycle')
            for i in range(1, len(gait_cycle_index)-1, 2):
                ans_list.append(video_tensor[gait_cycle_index[i]:gait_cycle_index[i+1], ...])
                use_idx.append(gait_cycle_index[i])

        logging.info(f"used split gait cycle index: {use_idx}")
        return ans_list, use_idx # needed gait cycle video tensor

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
        gait_cycle_index_bbox = file_info_dict["gait_cycle_index"]
        bbox_none_index = file_info_dict["none_index"]
        bbox = file_info_dict["bbox"]

        logging.info(f"video name: {video_name}, gait cycle index: {gait_cycle_index_bbox}")


        # use temporal mix or not
        # TODO: should return the new frame, named temporal mix.
        if self._temporal_mix:
            defined_vframes = self.temporal_mix(vframes, gait_cycle_index_bbox, bbox)

        else:
            # split gait by gait cycle index, first phase or second phase
            defined_vframes, used_gait_idx = self.split_gait_cycle(vframes, gait_cycle_index_bbox, self._gait_cycle)


        sample_info_dict = {
            "video": defined_vframes,
            "label": label,
            "disease": disease,
            "video_name": video_name,
            "video_index": index,
            "gait_cycle_index": gait_cycle_index_bbox,
            "bbox_none_index": bbox_none_index,
        }
        
        # move to functional
        if self._transform is not None:
            video_t_list = []
            for video_t in defined_vframes:
                video_t_list.append(self._transform(video_t.permute(1, 0, 2, 3)))

            sample_info_dict["video"] = torch.stack(video_t_list, dim=0) # c, t, h, w
        else:
            print("no transform")

        return sample_info_dict


def labeled_gait_video_dataset(
    gait_cycle: int,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset_idx: Dict = None,
    temporal_mix: bool = False,
) -> LabeledGaitVideoDataset:

    dataset = LabeledGaitVideoDataset(
        gait_cycle,
        dataset_idx,
        transform,
        temporal_mix,
    )

    return dataset
