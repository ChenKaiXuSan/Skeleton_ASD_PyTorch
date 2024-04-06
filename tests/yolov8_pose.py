"""
File: yolov8.py
Project: tests
Created Date: 2023-09-03 12:58:59
Author: chenkaixu
-----
Comment:
 
Have a good code time!
-----
Last Modified: Sunday September 3rd 2023 12:58:59 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

27-11-2023	Kaixu Chen	we change the shape of mask, keypoint and bbox.
2023-09-10	KX.C	we process whole batch in function.

"""

# %%
import os, logging, shutil, sys, math

sys.path.append("/workspace/skeleton")
import torch
from torchvision.transforms.functional import crop, pad, resize, hflip
from torchvision.utils import save_image, draw_keypoints, draw_bounding_boxes
from torchvision.io import read_image, read_video, write_video, write_png
from torchvision.ops import box_convert

from math import pi
import numpy as np

import hydra
import matplotlib.pyplot as plt

from prepare.prepare_gait_dataset.preprocess import Preprocess

from ultralytics import YOLO

# define keypoint in COCO dataset
categories_mapping = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# connect coco dataset keypoint
connect_skeleton = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]

# define the part used to calc the angle
ANGLE = ["nose", "shoulder", "hip", "knee", "ankle"]


def get_mid_keypoint(kpt: torch.Tensor, part: str):
    """
    get the mid keypoint of given categories, with left and right keypoint.

    Returns:
        dict: mid keypoint
    """

    k, xy = kpt.shape

    if part == "nose":
        return kpt[categories_mapping[part]]
    else:
        left = kpt[categories_mapping["left_" + part]]
        right = kpt[categories_mapping["right_" + part]]

        mid = (left + right) / 2

        return mid


def calc_angle(mid_angle_Dict: dict) -> torch.Tensor:
    ans_angle_List = []

    mid_angle_key = list(mid_angle_Dict.keys())
    for i in range(len(mid_angle_key) - 1):
        a = mid_angle_Dict[mid_angle_key[i]]
        x1, y1 = a.tolist()
        b = mid_angle_Dict[mid_angle_key[i + 1]]
        x2, y2 = b.tolist()

        vector_AB = torch.tensor([x2 - x1, y2 - y1])

        angle_radians = torch.atan2(vector_AB[1], vector_AB[0])

        print(angle_radians)

        # ! 坐标系有问题，还需要再调整
        angle_deg = torch.rad2deg(angle_radians)
        print(180 - angle_deg)

        # because the y axis is different for calc and keypoint, so need to change the angle
        ans_angle_List.append(180 - angle_deg)

    return torch.stack(ans_angle_List, dim=0)  # k, angle_deg


def get_spinal_angle(pose: torch.Tensor):
    b, t, k, xy = pose.shape

    batch_mid_List = []
    for batch in range(b):
        frame_mid_List = []
        frame_mid_angle_list = []

        for f in range(t):
            mid_List = []
            mid_angle_Dict = {}

            for angle in ANGLE:
                mid_coordinate = get_mid_keypoint(pose[batch, f], part=angle)
                mid_angle_Dict[angle] = mid_coordinate

                mid_List.append(mid_coordinate)

            # * calc the angle between the mid point of indicated part
            frame_mid_angle_list.append(calc_angle(mid_angle_Dict))

            # * frame mid list in time sequence
            frame_mid_List.append(
                torch.stack(mid_List, dim=0)
            )  # k, xy, k=5, xy is the mid coordinate

        batch_mid_List.append(torch.stack(frame_mid_List, dim=0))  # f, k, xy

    return torch.stack(batch_mid_List, dim=0)  # b, f, k, xy


def get_gait_cycle_from_pose(pose: torch.Tensor, part="left_ankle"):
    """
    gait_cycle, this method is used to find the gait cycle in the video.
    Use the left ankel point to find the gait cycle, calc the left distance between the left ankel point and the left edge of the frame.
    ! here will have some promble, that when the person direction is not the same, the gait cycle index will be wrong.
    ! so i think need to find another way to find the gait cycle index, such as use middle point as the reference point, to calc the gait cycle index.

    Args:
        pose (torch.Tensor): estimate pose from the video, shape is [b, c, k, xy]
        part (str, optional): wich part used to calc. Defaults to "left_ankle".

    Returns:
        Dict: gait cycle index, key is the frame index, value is the distance between the left ankel point and the left edge of the frame.
    """

    b, t, k, xy = pose.shape

    # the middle value in the frame for left ankel
    mid_value = 0.5
    distance = 0.01
    # the min value for left ankel point
    min_ankle_x = pose[..., categories_mapping[part], 0].min()

    ankel_x_Dict = {}

    for batch in range(b):
        for f in range(t):
            # define the x,y coordinate for left ankel
            ankle_x, ankle_y = pose[batch, f, categories_mapping[part]]

            # print(f, ankle_x, ankle_y)
            if ankle_x < mid_value and abs(ankle_x - min_ankle_x) < distance:
                ankel_x_Dict[f] = ankle_x

    # * find the min value in different frame region
    ans_Dict = {}
    # 使用 sorted 函数对字典按键进行排序
    sorted_data = sorted(ankel_x_Dict.items())

    # 初始化变量来跟踪当前区间的起始和结束索引
    start_index = None
    end_index = None

    # 初始化变量来跟踪最小值和最小值的索引
    min_value = None
    min_index = None

    # 遍历排序后的数据
    for index, (key, value) in enumerate(sorted_data):
        # 如果是第一个元素，将起始索引和最小值初始化为当前元素
        if start_index is None:
            start_index = index
            end_index = index
            min_value = value
            min_index = index
        else:
            # 如果当前元素的键与上一个元素的键相差大于1，则表示出现了不连续的区间
            if key - sorted_data[index - 1][0] > 1:
                # 输出当前区间的最小值
                # print(f'区间 {start_index}:{end_index} 最小值为 {sorted_data[min_index][0]}:{min_value}')
                ans_Dict[sorted_data[min_index][0]] = min_value

                # 重置变量以处理下一个区间
                start_index = index
                end_index = index
                min_value = value
                min_index = index
            else:
                # 如果当前元素的值比最小值小，则更新最小值和最小值的索引
                if value < min_value:
                    min_value = value
                    min_index = index
                # 更新当前区间的结束索引
                end_index = index

    # 输出最后一个区间的最小值
    if start_index is not None:
        # print(f'区间 {start_index}:{end_index} 最小值为 {sorted_data[min_index][0]}:{min_value}')
        ans_Dict[sorted_data[min_index][0]] = min_value

    return ans_Dict


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

    bias = 100  # pixel value

    min_width = 0
    max_width = float("inf")

    ans = []

    # first process one batch
    for batch in range(b):
        total_width = bbox[batch, 2]

        min_width = total_width.min()
        max_width = total_width.max()

        for f in range(t):
            # define the x,y coordinate for left ankel
            x, y, w, h = bbox[batch, f]

            if abs(w - max_width) < bias:
                ans.append(f)

    # return the max width frame index
    return ans


def _draw_keypoints(kpt: torch.Tensor, video: torch.Tensor):
    b, c, t, h, w = video.shape
    b, f, k, xy = kpt.shape
    # connect the different keypoint, which nose, mid of shoulder, mid of hip, mid of knee, mid of ankle
    connect_skeleton = [(0, 1), (1, 2), (2, 3), (3, 4)]

    ans_video = []

    for frame in range(t):
        res = draw_keypoints(
            video[0, :, frame],
            kpt[:, frame, ...] * h,
            radius=5,
            colors="blue",
            connectivity=connect_skeleton,
        )

        ans_video.append(res.permute(1, 2, 0))
        write_png(res, f"/workspace/skeleton/tests/test_{frame}.png")

    write_video(
        video_array=torch.stack(ans_video),
        filename="/workspace/skeleton/tests/test.mp4",
        fps=30,
    )

    logging.info("save the figure")


def _draw_ankle_curve(kpt: torch.Tensor, part="left_ankle"):
    """
    _draw_ankle_curve, draw the ankle curve from video, reference with the indicated part.
    x axis is the frame index, y axis is the x coordinate of the indicated part point.

    Args:
        kpt (torch.Tensor): the predicted keypoint from the video, shape is [b, t, k, xy]
        video (torch.Tensor): video, shape is [b, c, t, h, w]
    """

    b, f, k, xy = kpt.shape

    y_list = []
    for frame in range(f):
        ankle_x, ankle_y = kpt[0, frame, categories_mapping[part]]
        y_list.append(ankle_x.cpu().numpy())

    x = np.arange(0, f, 1)

    # draw the ankle curve
    plt.figure()
    plt.plot(x, y_list, "o")

    plt.title("ankle curve")
    plt.xlabel("frames")
    plt.ylabel("ankle x coordinate")
    plt.xlim(0, f)

    plt.savefig(f"/workspace/skeleton/tests/ankle_curve.png")
    # plt.legend()
    plt.show()

    logging.info("draw the ankle curve and save it!")


def _draw_overlap_bbox(bbox: torch.Tensor, video: torch.Tensor):
    b, c, t, height, width = video.shape
    b, t, xywh = bbox.shape

    # convert the bbox from xywh to xyxy
    xyxy_bbox = box_convert(bbox[0], in_fmt="cxcywh", out_fmt="xyxy")

    # draw the bbox
    drawn_boxes = draw_bounding_boxes(video[0, :, 0], xyxy_bbox[:, :].squeeze())

    # save the drawn boxes
    plt.figure()
    plt.imshow(drawn_boxes.permute(1, 2, 0))
    plt.title("stack whole frames bbox")

    plt.legend()
    plt.show()
    plt.savefig(f"/workspace/skeleton/tests/bbox.png")
    plt.figure().clear()


def _draw_bbox_curve(bbox: torch.Tensor):
    b, t, xywh = bbox.shape

    y_list = [i for i in range(t)]

    x_list = []

    center = 512 // 2

    # only draw one batch
    bbox = bbox[0]

    for frame in range(t):
        x, y, w, h = bbox[frame]
        x_list.append(
            [abs(w.cpu().numpy() // 2 - center), w.cpu().numpy() // 2 + center]
        )

    # 创建图表
    plt.figure()
    for i, x_row in enumerate(x_list):
        plt.plot(x_row, [y_list[i]] * len(x_row), label=f"Line {i+1}")

    plt.xlim(0, 512)
    plt.xlabel("width of whole bbox")
    plt.ylabel("frame")

    # plt.legend()
    plt.show()
    plt.savefig(f"/workspace/skeleton/tests/bbox_width_curve.png")


@hydra.main(config_path="/workspace/skeleton/configs/", config_name="config")
def init_params(config):
    # DATA_PATH = config.data_path
    # SAVE_PATH = config.save_path
    IMG_SIZE = config.data.img_size
    PATH = "/workspace/data/split_pad_dataset_512/flod0/train/ASD/20160426_ASD_lat__V1-0001.mp4"
    # PATH = "/workspace/data/split_pad_dataset_512/flod0/train/ASD/20160426_ASD_lat__V1-0002.mp4"

    preprocess = Preprocess(config)
    video, _, info = read_video(PATH, pts_unit="sec", output_format="TCHW")

    # if right to left, need to flip the video
    # video = hflip(video)

    video = video.unsqueeze(0)
    video = video.permute(0, 2, 1, 3, 4)  # b,c,t,h,w

    # * return value shape:
    # batch, (b, c, t, h, w)
    # label, (b)
    # bbox, (b, t, 4) (cxcywh)
    # mask, (b, 1, t, h, w)
    # keypoint, (b, t, 17, 2)
    video, label, optical_flow, bbox, mask, pose = preprocess(
        video, torch.tensor([1]), 0
    )

    # * prediction cycle from pose
    gait_cycle_index_pose = get_gait_cycle_from_pose(pose, part="left_ankle")

    # * prediction cycle from bbox
    _draw_bbox_curve(bbox)
    gait_cycle_index_bbox = get_gait_cycle_from_bbox(bbox)
    _draw_overlap_bbox(bbox, video)

    mid_coord = get_spinal_angle(
        pose
    )  # b, f, k, xy, here k=5, xy is the mid coordinate
    print(mid_coord.shape)

    # _draw_keypoints(mid_coord, video)

    # _draw_ankle_curve(pose, part='left_ankle')


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()