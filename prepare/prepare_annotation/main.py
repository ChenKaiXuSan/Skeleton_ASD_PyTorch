"""
File: main.py
Project: prepare_annotation
Created Date: 2023-10-20 08:13:48
Author: chenkaixu
-----
Comment:
This file is used for prepare the annotation file for training.
In different step:
1. load the video path.
2. pose estimation.
3. define the gait cycle.
4. save the annotation file.
5. save the static images.
 
Have a good code time!
-----
Last Modified: 2023-10-26 05:33:25
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

"""

from typing import Any
import torch
from pathlib import Path
from torchvision.transforms.functional import crop, pad, resize
from torchvision.utils import save_image
from torchvision.io import read_image, read_video, write_video, write_png

import os, shutil, sys, multiprocessing, logging
from argparse import ArgumentParser

import hydra
import matplotlib.pyplot as plt

from project.utils import del_folder, make_folder

from prepare_annotation.videos_to_frames import *
from prepare_annotation.one_gait_cycle import define_gait_cycle

# from prepare_annotation.process_annotation_file import

CLASS = ["ASD", "DHS", "LCS_HipOA", "Normal"]
VIEW = ["right", "left", "front", "back"]


def get_parameters():
    """
    get_parameters from python script.

    Returns:
        parser: include parsers.
    """
    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument("--img_size", type=int, default=512)

    # Path
    parser.add_argument(
        "--data_path",
        type=str,
        default="/workspace/data/four_view_data_512",
        help="meta four view dataset path, with human annotation.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/workspace/data/annotated_four_view_data_512",
        help="convert the four view dataset into a format of the annotation file.",
    )

    return parser.parse_known_args()


class LoadOneDisese:
    def __init__(self, data_path, disease) -> None:
        self.DATA_PATH = data_path
        self.disease = disease
        self.path_dict = {}

    def process_one_patient(self, one_patient: Path):
        # FIXME some patient not have 4 view.
        for view in one_patient.iterdir():
            self.path_dict[view.name] = view # {view: view_path}

    def process_one_class(self, one_class: Path):
        for one_file in one_class.iterdir():
            if "finish" in one_file.name:
                print(one_file)
                self.process_one_patient(one_file)

    def process_one_disease(self):
        for flag in self.disease.split("_"):
            raw_data_path = Path(self.DATA_PATH, flag)
            for one_class in raw_data_path.iterdir():
                # delete the data.xls
                if "convert.py" in one_class.name:
                    continue
                else:
                    print(one_class)
                    self.process_one_class(one_class)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.process_one_disease()
        return self.path_dict


def process(parames, disease: str):

    # DISEASE = ['ASD', 'DHS', 'LCS', 'HipOA']
    DATA_PATH = parames.data_path
    SAVE_PATH = parames.save_path

    # step1: resort the number of the img, and split into different folder.
    # * load the folder
    load_one_disease = LoadOneDisese(DATA_PATH, disease)
    video_path_dict = load_one_disease()  # {view: view_path}

    # step2: define one gait cycle
    define_gait_cycle(parames, video_path_dict)
    # step3: save the annotation file


@hydra.main(config_path="/workspace/skeleton/configs/", config_name="prepare_dataset")
def main(parames):
    """
    main, for the multiprocessing.

    Args:
        parames (hydra): hydra config.
    """

    # define process
    # ASD_process = multiprocessing.Process(target=main, args=(parames, 'ASD'))
    # DHS_process = multiprocessing.Process(target=main, args=(parames, 'DHS'))
    # LCS_process = multiprocessing.Process(target=main, args=(parames, "LCS_HipOA"))
    # # HipOA_process = multiprocessing.Process(target=main, args=(parames, 'normal'))

    # # start process
    # # ASD_process.start()
    # # DHS_process.start()
    # LCS_process.start()
    # # HipOA_process.start()

    process(parames, "LCS_HipOA")


if __name__ == "__main__":
    # parames, unknown = get_parameters()

    # ! just used for debug
    main()
