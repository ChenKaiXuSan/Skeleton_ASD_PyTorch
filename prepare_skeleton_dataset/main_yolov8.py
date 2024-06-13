#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/skeleton/prepare_gait_cycle_index/main_yolov8.py
Project: /workspace/skeleton/prepare_gait_cycle_index
Created Date: Thursday June 13th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday June 13th 2024 1:21:41 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

from __future__ import annotations
from typing import Dict, List, Tuple

import pickle
import torch
from pathlib import Path
from torchvision.io import read_video, write_png

import os, shutil, sys, multiprocessing, logging, json, threading
import mmengine

sys.path.append("/workspace/skeleton/")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import hydra

from project.utils import del_folder, make_folder
from prepare_gait_cycle_index.preprocess import Preprocess

RAW_CLASS = ["ASD", "DHS", "LCS", "HipOA"]
CLASS = ["ASD", "DHS", "LCS_HipOA", "Normal"]
map_CLASS = {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "Normal": 3}  # map the class to int

def gait_cycle_index_visualization(
    frames: torch.Tensor,
    gait_cycle_index: list,
    keypoint: torch.Tensor,
    bbox: torch.Tensor,
):
    pass


def process_bbox(bbox: torch.Tensor) -> List:
    # find min and max bbox weight
    b, t, xywh = bbox.shape

    # * get the gait cycle index with bbox width
    bias = 50  # pixel value
    threshold = 10

    min_width = 0
    max_width = float("inf")

    ans = []

    # first process one batch
    for batch in range(b):
        total_width = bbox[batch, :, 2]

        min_width = total_width.min()
        max_width = total_width.max()

        for f in range(t):
            # define the x,y coordinate for left ankel
            x, y, w, h = bbox[batch, f]

            if abs(w - max_width) < bias:
                ans.append(f)

    # filter close values
    bbox_res = [0]

    for i in range(len(ans)):
        if not bbox_res:
            bbox_res.append(ans[i])
        else:
            if abs(ans[i] - bbox_res[-1]) > threshold:
                bbox_res.append(ans[i])

    bbox_res.append(t)  # add the last frame index

    # return the max width frame index
    return bbox_res


def process_keypoint(pose: torch.Tensor, frames: torch.Tensor) -> List:

    # * get the gait cycle index with pose keypoint
    b, t, point_num, coordinate = pose.shape
    b, c, t, h, w = frames.shape

    bias = 50  # pixel value
    threshold = 10

    foot_max_width = float("-inf")
    foot_ans = []

    for batch in range(b):
        # b, t, 17, 2
        # find the max width of foot coordinate
        for f in range(t):
            foot_coordinate = pose[batch, f, -2:].t()  # x, y > y, x

            # fine the max width of foot width
            width = abs(foot_coordinate[0][0] - foot_coordinate[0][1]) * h
            if width > foot_max_width:
                foot_max_width = width

        # find the min bias between the foot width and the max width
        # it means the index is the gait cycle index
        for f in range(t):

            foot_coordinate = pose[batch, f, -2:].t()  # x, y > y, x
            foot_width = abs(foot_coordinate[0][0] - foot_coordinate[0][1]) * h

            if abs(foot_width - foot_max_width) < bias:
                foot_ans.append(f)

    # filter close values
    foot_res = [0]

    for i in range(len(foot_ans)):
        if not foot_res:
            foot_res.append(foot_ans[i])
        else:
            if abs(foot_ans[i] - foot_res[-1]) > threshold:
                foot_res.append(foot_ans[i])

    # return the index by foot width
    return foot_res


def get_gait_cycle_from_bbox_keypoint(
    frames: torch.Tensor, bbox: torch.Tensor, pose: torch.Tensor
) -> List:
    """predict the gait cycle from bbox.
    use the max width bbox as the gait cycle frame index.
    The motivation is that the max width bbox is the most complete bbox in the video.

    Args:
        bbox (torch.Tensor): bbox from the video, shape is [b, t, xywh]

    Returns:
        list: the gait cycle frame index,.
    """

    bbox_gait_index = process_bbox(bbox)
    keypoint_gait_index = process_keypoint(pose, frames)

    # compare the final result between bbox and keypoint
    if len(bbox_gait_index) == len(keypoint_gait_index):
        final_gait_index = bbox_gait_index
    elif len(bbox_gait_index) > len(keypoint_gait_index):
        final_gait_index = bbox_gait_index
    elif len(bbox_gait_index) < len(keypoint_gait_index):
        final_gait_index = keypoint_gait_index

    return final_gait_index

class LoadOneDisese:
    def __init__(self, data_path, fold, disease) -> None:

        self.DATA_PATH = data_path
        self.fold = fold
        self.disease = disease
        # this is the return dict
        self.path_dict = {key: [] for key in CLASS}

    def process_class(self, one_class: Path, flag: str):
        for d in self.disease:
            if d in ["DHS", "LCS", "HipOA"]:
                for one_video in sorted(one_class.iterdir()):
                    if d in one_video.name.split("_"):
                        info = {
                            "flag": flag,
                            "disease": d,
                        }

                        if d in CLASS:
                            self.path_dict[d].append((str(one_video), info))
                        else:
                            self.path_dict[CLASS[2]].append((str(one_video), info))
                    else:
                        continue

            else:
                for one_video in sorted(one_class.iterdir()):
                    if d in one_video.name.split("_"):
                        info = {"flag": flag, "disease": d}

                        self.path_dict[d].append((str(one_video), info))

    def __call__(self) -> dict:
        one_fold_path = Path(self.DATA_PATH, self.fold)

        for flag in ["train", "val"]:
            one_flag_path = Path(one_fold_path, flag)

            for one_class in self.disease:
                if one_class == "DHS" or one_class == "LCS" or one_class == "HipOA":
                    new_path = one_flag_path / "ASD_not"
                else:
                    new_path = one_flag_path / one_class

                self.process_class(new_path, flag)

        return self.path_dict


def save_to_json(sample_info, save_path, logger, method: str) -> None:
    """save the sample info to json file.

    There have three method to get the gait cycle index, include mix, pose, bbox.

    The sample info include:

    video_name: the video name,
    video_path: the video path, relative path from /workspace/skeleton/data/segmentation_dataset_512
    frame_count: the raw frames of the video,
    label: the label of the video,
    disease: the disease of the video,
    gait_cycle_index: the gait cycle index,
    bbox_none_index: the bbox none index, when use yolo to get the bbox, some frame will not get the bbox.
    bbox: the bbox, [n, 4] (cxcywh)

    Args:
        sample_info (dict): the sample info dict.
        save_path (str): the prefix save path, like /workspace/skeleton/data/segmentation_dataset_512
        logger (Logger): for multiprocessing logging.
        method (str): the method to get the gait cycle index, include mix, pose, bbox.
    """

    save_path = Path(save_path, "json_" + method)

    save_path_with_name = (
        save_path / sample_info["disease"] / (sample_info["video_name"] + ".json")
    )

    make_folder(save_path_with_name.parent)
    with open(save_path_with_name, "w") as f:
        json.dump(sample_info, f, indent=4)
    logger.info(f"Save the {sample_info['video_name']} to {save_path}")


def process(parames, fold: str, disease: list):
    DATA_PATH = parames.gait_dataset.data_path
    SAVE_PATH = parames.gait_dataset.save_path

    res = {'train': [], 'val': []}

    # prepare the log file
    logger = logging.getLogger(f"Logger-{multiprocessing.current_process().name}")

    logger.info(f"Start process the {fold} dataset")

    # prepare the preprocess
    preprocess = Preprocess(parames)

    # * step1: load the video path, with the sorted order
    load_one_disease = LoadOneDisese(DATA_PATH, fold, disease)
    one_fold_video_path_dict = (
        load_one_disease()
    )  # {disease: (video_path, info{flag, disease}})}

    # k is disease, v is (video_path, info)
    for k, v in one_fold_video_path_dict.items():
        if v:
            logger.info(f"Start process the {k} disease")

        for video_path, info in v:
            # * step2: load the video from vieo path
            # get the bbox
            vframes, audio, _ = read_video(
                video_path, pts_unit="sec", output_format="TCHW"
            )

            # * step3: use preprocess to get information.
            # the format is: final_frames, bbox_none_index, label, optical_flow, bbox, mask, pose

            label = torch.tensor([map_CLASS[k]])  # convert the label to int
            # TCHW > CTHW > BCTHW
            m_vframes = vframes.permute(1, 0, 2, 3).unsqueeze(0)
            (
                frames,
                bbox_none_index,
                label,
                optical_flow,
                bbox,
                mask,
                keypoints,
                keypoints_score
            ) = preprocess(m_vframes, label, 0)

            # * step4: save the video frames keypoint 
            anno = dict()

            anno['keypoint'] = keypoints
            anno['keypoint_score'] = keypoints_score
            anno['frame_dir'] = video_path
            anno['img_shape'] = (vframes.shape[2], vframes.shape[3])
            anno['original_shape'] = (vframes.shape[2], vframes.shape[3])
            anno['total_frames'] = keypoints.shape[1]
            anno['label'] = int(label)

            res[info['flag']].append(anno)
            # break;
    mmengine.dump(res, str(SAVE_PATH) + f'/{"_".join(disease)}.pkl')
    
def merge_pkl(config):
    
    anns = {
        'annotations': []
    }
    split_dict = {
        'train': [],
        'val': []
    }

    SAVE_PATH = Path(config.gait_dataset.save_path)

    for disease in SAVE_PATH.iterdir():


        with open(disease, 'rb') as file:
            data = pickle.load(file)

        train_anns = data['train']
        for one_sample in train_anns:
            frame_dir = one_sample['frame_dir']
            split_dict['train'].append(frame_dir)
            anns['annotations'].append(one_sample)

        val_anns = data['val']
        for one_sample in val_anns:
            frame_dir = one_sample['frame_dir']
            split_dict['val'].append(frame_dir)
            anns['annotations'].append(one_sample)

    anns["split"] = split_dict
    with open(SAVE_PATH / 'whole_annotations.pkl', 'wb') as file:
        pickle.dump(anns, file)

@hydra.main(config_path="/workspace/skeleton/configs/", config_name="prepare_skeleton_dataset")
def main(parames):
    """
    main, for the multiprocessing using.

    Args:
        parames (hydra): hydra config.
    """

    # ! only for test
    # process(parames, "fold0", ["ASD"])

    threads = [] 
    for d in [["ASD"], ["LCS", "HipOA"]]:

        thread = multiprocessing.Process(target=process, args=(parames, "fold0", d))
        threads.append(thread)

    parames.YOLO.device = "cuda:1"    
    threads.append(multiprocessing.Process(target=process, args=(parames, "fold1", ["DHS"])))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    merge_pkl(parames)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    main()
