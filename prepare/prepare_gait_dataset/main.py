"""
File: main.py
Project: prepare_gait_dataset
Created Date: 2023-10-30 06:50:52
Author: chenkaixu
-----
Comment:
This script is used to the segmentation_dataset_512.
To define the gait cycle in the video, and save the splitted frame in video.
 
Have a good code time!
-----
Last Modified: 2023-10-30 06:51:12
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

"""
from __future__ import annotations

from typing import Any
import torch
from pathlib import Path
from torchvision.transforms.functional import crop, pad, resize
from torchvision.utils import save_image
from torchvision.io import read_image, read_video, write_video, write_png

import os, shutil, sys, multiprocessing, logging
sys.path.append("/workspace/skeleton/")
from argparse import ArgumentParser

import hydra
import matplotlib.pyplot as plt

from project.utils import del_folder, make_folder

from prepare.prepare_gait_dataset.preprocess import Preprocess

RAW_CLASS = ["ASD", "DHS", "LCS", "HipOA"]
CLASS = ["ASD", "DHS", "LCS_HipOA", "Normal"]
map_CLASS = {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "Normal": 3}
VIEW = ["right", "left", "front", "back"]
FOLD = ["fold0", "fold1", "fold2", "fold3", "fold4"]


def get_gait_cycle_from_bbox(bbox: torch.Tensor):
    """predict the gait cycle from bbox.
    use the max width bbox as the gait cycle frame index.
    The motivation is that the max width bbox is the most complete bbox in the video.

    Args:
        bbox (torch.Tensor): bbox from the video, shape is [b, t, xywh]

    Returns:
        list: the gait cycle frame index,.
    """

    # find min and max bbox weight
    b, t, xywh = bbox.shape

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
    res = [0] 

    for i in range(len(ans)):
        if not res:
            res.append(ans[i])
        else:
            if abs(ans[i] - res[-1]) > threshold:
                res.append(ans[i])

    # return the max width frame index
    return res


class LoadOneDisese:
    def __init__(self, data_path, fold) -> None:
        self.DATA_PATH = data_path
        self.fold = fold
        self.path_dict = {key:[] for key in CLASS}

    def process_class(self, one_class: Path, flag: str, disease: str):

        if disease == "ASD_not":
            for one_video in one_class.iterdir():
                
                for raw_disease in RAW_CLASS:
                    if raw_disease in one_video.name.split("_"):
                        
                        info = {"flag": flag, "disease": raw_disease, }
                        break; # if find the disease, break the loop.

                if raw_disease in CLASS:
                    self.path_dict[raw_disease].append((str(one_video), info))
                else:
                    self.path_dict[CLASS[2]].append((str(one_video), info))

        else:
            for one_video in one_class.iterdir():

                for raw_disease in RAW_CLASS:
                    if raw_disease in one_video.name.split("_"):
                        info = {"flag": flag, "disease": raw_disease}
                        break;

                self.path_dict[disease].append((str(one_video), info) )

    def process_one_fold(self):
        one_fold_path = Path(self.DATA_PATH, self.fold)

        for flag in ["train", "val"]:
            one_flag_path = Path(one_fold_path, flag)

            for one_class in one_flag_path.iterdir():
                disease = one_class.name
                self.process_class(one_class, flag, disease)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.process_one_fold()
        return self.path_dict

def process(parames, fold: str):
    # DISEASE = ['ASD', 'DHS', 'LCS', 'HipOA']
    DATA_PATH = parames.gait_dataset.data_path
    SAVE_PATH = parames.gait_dataset.save_path

    # prepare the preprocess 
    preprocess = Preprocess(parames)
    # step1: resort the number of the img, and split into different folder.
    # * load the folder
    load_one_disease = LoadOneDisese(DATA_PATH, fold)
    one_fold_video_path_dict = load_one_disease()  # {disease: (video_path, info{flag, disease}})}

    for k, v in one_fold_video_path_dict.items():
        for video_path, info in v:
            # get the bbox
            # * step2: load the video from vieo path 
            vframes, audio, _ = read_video(video_path, pts_unit="sec", output_format="TCHW")
            # * step3: use preprocess to get the bbox
            label = torch.tensor([map_CLASS[k]]) # convert the label to int
            # TCHW > CTHW > BCTHW
            m_vframes = vframes.permute(1,0,2,3).unsqueeze(0)
            _, label, optical_flow, bbox, mask, pose = preprocess(m_vframes, label, 0)

            # * step4: use bbox to define the gait cycle, and get the gait cycle index
            gait_cycle_index_bbox = get_gait_cycle_from_bbox(bbox)
            # * step5: use gait cycle index to split the video
            frames = []
            for i in range(0, len(gait_cycle_index_bbox)-2, 2):
                frames.append(vframes[
                    gait_cycle_index_bbox[i]:gait_cycle_index_bbox[i+1]
                ].permute(1,0,2,3)
                )
            # * step6: save the video frames to pt file
            save_tensor_dict = {
                "video": frames,
                "label": label,
                "disease": k,
                "gait_cycle_index": gait_cycle_index_bbox,
            }

            save_path = Path(SAVE_PATH, fold, info['flag'], info['disease'])
            make_folder(save_path)
            file_name = video_path.split('/')[-1].split('.')[0] + '.pt'
            torch.save(save_tensor_dict, str(save_path.joinpath(file_name)))

@hydra.main(config_path="/workspace/skeleton/configs/", config_name="prepare_dataset")
def main(parames):
    """
    main, for the multiprocessing.

    Args:
        parames (hydra): hydra config.
    """

    # define process
    fold0 = multiprocessing.Process(target=process, args=(parames, 'fold0'))
    fold1 = multiprocessing.Process(target=process, args=(parames, 'fold1'))
    fold2 = multiprocessing.Process(target=process, args=(parames, 'fold2'))
    fold3 = multiprocessing.Process(target=process, args=(parames, 'fold3'))
    fold4 = multiprocessing.Process(target=process, args=(parames, 'fold4'))

    # start process 
    fold0.start()
    fold1.start()
    fold2.start()
    fold3.start()
    fold4.start()

if __name__ == "__main__":
    # parames, unknown = get_parameters()

    # ! just used for debug
    main()
