"""
File: main.py
Project: prepare_gait_dataset
Created Date: 2023-10-30 06:50:52
Author: chenkaixu
-----
Comment:
This script is used to the segmentation_dataset_512 dataset.
To define the gait cycle in the video, and save the gait cycle index to json file.

The json file include: (in dict)
    video_name: the video name,
    video_path: the video path, relative path from /workspace/skeleton/data/segmentation_dataset_512
    frame_count: the raw frames of the video,
    label: the label of the video,
    disease: the disease of the video,
    gait_cycle_index_bbox: the gait cycle index,
    bbox_none_index: the bbox none index, when use yolo to get the bbox, some frame will not get the bbox.
    bbox: the bbox, [n, 4] (cxcywh)

Have a good code time!
-----
Last Modified: 2023-10-30 06:51:12
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

14-12-2023	Kaixu Chen	recode the code, and delete the useless code.

12-12-2023	Kaixu Chen	the .pt file is too large, so change to use the json file to save the bbox and gait cycle index.

"""
from __future__ import annotations

import torch
from pathlib import Path
from torchvision.io import read_image, read_video, write_video, write_png

import os, shutil, sys, multiprocessing, logging, json
sys.path.append("/workspace/skeleton/")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import hydra

from project.utils import del_folder, make_folder
from preprocess import Preprocess

RAW_CLASS = ["ASD", "DHS", "LCS", "HipOA"]
CLASS = ["ASD", "DHS", "LCS_HipOA", "Normal"]
map_CLASS = {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "Normal": 3}  # map the class to int
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

    res.append(t)  # add the last frame index
    # return the max width frame index
    return res


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

def save_to_json(sample_info, save_path, logger):
    """save the sample info to json file.

    the sample info include:
    video_name: the video name,
    video_path: the video path, relative path from /workspace/skeleton/data/segmentation_dataset_512
    frame_count: the raw frames of the video,
    label: the label of the video,
    disease: the disease of the video,
    gait_cycle_index_bbox: the gait cycle index,
    bbox_none_index: the bbox none index, when use yolo to get the bbox, some frame will not get the bbox.
    bbox: the bbox, [n, 4] (cxcywh)

    Args:
        sample_info (dict): the sample info dict.
        save_path (str): the prefix save path, like /workspace/skeleton/data/segmentation_dataset_512
        logger (Logger): for multiprocessing logging.
    """

    save_path = Path(save_path, "json")

    save_path_with_name = (
        save_path / sample_info["disease"] / (sample_info["video_name"] + ".json")
    )
    make_folder(save_path_with_name.parent)
    with open(save_path_with_name, "w") as f:
        json.dump(sample_info, f)
    logger.info(f"Save the {sample_info['video_name']} to {save_path}")


def process(parames, fold: str, disease: list):
    DATA_PATH = parames.gait_dataset.data_path
    SAVE_PATH = parames.gait_dataset.save_path

    # prepare the log file
    logger = logging.getLogger(f"Logger-{multiprocessing.current_process().name}")

    handler = logging.FileHandler(Path(parames.log_path, f"{fold}_{disease}.log"))
    logger.addHandler(handler)

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
            # * step3: use preprocess to get the bbox
            label = torch.tensor([map_CLASS[k]])  # convert the label to int
            # TCHW > CTHW > BCTHW
            m_vframes = vframes.permute(1, 0, 2, 3).unsqueeze(0)
            (
                final_frames,
                bbox_none_index,
                label,
                optical_flow,
                bbox,
                mask,
                pose,
            ) = preprocess(m_vframes, label, 0)

            # * step4: use bbox to define the gait cycle, and get the gait cycle index
            gait_cycle_index_bbox = get_gait_cycle_from_bbox(bbox)

            # * step5: save the video frames to json file
            sample_json_info = {
                "video_name": video_path.split("/")[-1].split(".")[0],
                "video_path": video_path,
                "frame_count": vframes.shape[0],
                "label": int(label),
                "disease": k,
                "gait_cycle_index_bbox": gait_cycle_index_bbox,
                "bbox_none_index": bbox_none_index,
                "bbox": [bbox[0, i].tolist() for i in range(bbox.shape[1])],
            }

            save_to_json(sample_json_info, DATA_PATH, logger)


@hydra.main(config_path="/workspace/skeleton/configs/", config_name="prepare_dataset")
def main(parames):
    """
    main, for the multiprocessing using.

    Args:
        parames (hydra): hydra config.
    """

    # make the log save path
    del_folder(parames.log_path)
    make_folder(parames.log_path)

    # define process
    parames.YOLO.device = "cuda:0"
    asd = multiprocessing.Process(target=process, args=(parames, "fold0", ["ASD"]))
    asd.start()
    LCS_HipOA = multiprocessing.Process(
        target=process, args=(parames, "fold0", ["LCS", "HipOA"])
    )
    LCS_HipOA.start()

    parames.YOLO.device = "cuda:1"
    dhs = multiprocessing.Process(target=process, args=(parames, "fold0", ["DHS"]))
    dhs.start()

    # ! only for test
    # process(parames, "fold0", ["ASD"])


if __name__ == "__main__":
    main()
