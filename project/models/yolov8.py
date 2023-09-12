'''
File: yolov8.py
Project: models
Created Date: 2023-09-03 13:44:23
Author: chenkaixu
-----
Comment:
 
Have a good code time!
-----
Last Modified: 2023-09-10 06:42:03
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

'''


import torch
from torchvision.io import write_png

import numpy as np

import logging

from ultralytics import YOLO


class MultiPreprocess(torch.nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()

        # load model
        self.yolo_pose = YOLO(configs.pose_ckpt)
        self.yolo_mask = YOLO(configs.seg_ckpt)

        self.conf = configs.conf
        self.iou = configs.iou
        self.verbose = configs.verbose

    def get_YOLO_pose_result(self, frame_batch: np.ndarray):
        """
        get_YOLO_pose_result, from frame_batch, which (t, h, w, c)

        Args:
            frame_batch (np.ndarray): for processed frame batch, (t, h, w, c)

        Returns:
            dict, list: two return value, with one batch keypoint in Dict, and none index in list.
        """

        t, h, w, c = frame_batch.shape

        one_batch_keypoint = {}
        none_index = []

        with torch.no_grad():
            for frame in range(t):

                results = self.yolo_pose(
                    source=frame_batch[frame],
                    conf=self.conf,
                    iou=self.iou,
                    save_crop=False,
                    classes=0,
                    vid_stride=True,
                    stream=False,
                    verbose=self.verbose,
                )


                for i, r in enumerate(results):
                    # judge if have keypoints.
                    # one_batch_keypoint.append(r.keypoints.data) # 1, 17, 3
                    if list(r.keypoints.xyn.shape) != [1, 17, 2]:
                        none_index.append(frame)
                        one_batch_keypoint[frame] = None
                    else:
                        one_batch_keypoint[frame] = r.keypoints.xyn  # 1, 17

        return one_batch_keypoint, none_index

    def get_YOLO_mask_result(self, frame_batch: np.ndarray):
        """
        get_YOLO_mask_result, from frame_batch, for mask.

        Args:
            frame_batch (np.ndarry): for processed frame batch, (t, h, w, c)

        Returns:
            dict, list: two return values, with one batch mask in Dict, and none index in list.
        """

        t, h, w, c = frame_batch.shape

        one_batch_mask = {}
        none_index = []

        with torch.no_grad():
            for frame in range(t):
                results = self.yolo_mask(
                    source=frame_batch[frame],
                    conf=self.conf,
                    iou=self.iou,
                    save_crop=False,
                    classes=0,
                    vid_stride=True,
                    stream=False,
                    verbose=self.verbose,
                )


                for i, r in enumerate(results):
                    # judge if have mask.
                    if r.masks is None:
                        none_index.append(frame)
                        one_batch_mask[frame] = None
                    elif list(r.masks.data.shape) == [1, 224, 224]:
                        one_batch_mask[frame] = r.masks.data  # 1, 224, 224
                    else:
                        # when mask > 2, just use the first mask.
                        # ? sometime will get two type for masks.
                        one_batch_mask[frame] = r.masks.data[:1, ...]  # 1, 224, 224

        return one_batch_mask, none_index

    def delete_tensor(self, video: torch.tensor, delete_idx: int, next_idx: int):
        """
        delete_tensor, from video, we delete the delete_idx tensor and insert the next_idx tensor.

        Args:
            video (torch.tensor): video tensor for process.
            delete_idx (int): delete tensor index.
            next_idx (int): insert tensor index.

        Returns:
            torch.tensor: deleted and processed video tensor.
        """

        c, t, h, w = video.shape
        left = video[:, :delete_idx, ...]
        right = video[:, delete_idx + 1 :, ...]
        insert = video[:, next_idx, ...].unsqueeze(dim=1)

        ans = torch.cat([left, insert, right], dim=1)

        # check frame
        assert ans.shape[1] == t
        return ans

    def process_none(self, batch_Dict: dict, none_index: list):
        """
        process_none, where from batch_Dict to instead the None value with next frame tensor (or froward frame tensor).

        Args:
            batch_Dict (dict): batch in Dict, where include the None value when yolo dont work.
            none_index (list): none index list map to batch_Dict, here not use this.

        Returns:
            list: list include the replace value for None value.
        """

        boundary = len(batch_Dict) - 1  # 8

        for k, v in batch_Dict.items():
            if v == None:
                if (
                    None in list(batch_Dict.values())[k:]
                    and len(set(list(batch_Dict.values())[k:])) == 1
                ):
                    next_idx = k - 1
                else:
                    next_idx = k + 1
                    while batch_Dict[next_idx] == None and next_idx < boundary:
                        next_idx += 1

                batch_Dict[k] = batch_Dict[next_idx]

        return list(batch_Dict.values())

    def process_batch(self, batch: torch.Tensor, labels: list):
        b, c, t, h, w = batch.shape

        # for one batch prepare.
        pred_mask_list = []
        pred_keypoint_list = []

        for batch_index in range(b):

            # ! now, the ultralytics support torch.tensor type, but here have some strange problem. So had better use numpy type.
            # ! np.ndarray type, HWC format with BGR channels uint8 (0-255).
            # c, h, w > h, w, c, RGB > BGR
            one_batch_numpy = batch[batch_index, [2,1,0], ...].permute(1,2,3,0).to(torch.uint8).numpy()

            # check shape and dtype in numpy
            assert one_batch_numpy.shape == (t, h, w, c)
            assert one_batch_numpy.dtype == np.uint8

            # * process one batch mask
            one_batch_mask_Dict, one_mask_none_index = self.get_YOLO_mask_result(
                one_batch_numpy
            )
            one_batch_mask = self.process_none(one_batch_mask_Dict, one_mask_none_index)
            
            # * process one batch keypoint
            one_batch_keypoint_Dict, one_pose_none_index = self.get_YOLO_pose_result(
                one_batch_numpy
            )
            one_batch_keypoint = self.process_none(
                one_batch_keypoint_Dict, one_pose_none_index
            )

            pred_mask_list.append(torch.stack(one_batch_mask, dim=1))  # c, t, h, w
            pred_keypoint_list.append(
                torch.stack(one_batch_keypoint, dim=1)
            )  # c, t, keypoint, value

        # return batch, label, mask, keypoint
        return (
            batch,
            labels,
            torch.stack(pred_mask_list, dim=0),
            torch.stack(pred_keypoint_list, dim=0),
        )

    def forward(self, batch, labels):
        (
            b,
            c,
            t,
            h,
            w,
        ) = batch.shape

        # batch, (b, c, t, h, w)
        # label, (b)
        # mask, (b, 1, t, h, w)
        # keypoint, (b, 1, t, 17, 2)
        video, labels, mask, keypoint = self.process_batch(batch, labels)

        # shape check
        assert video.shape == batch.shape
        assert labels.shape[0] == b
        assert mask.shape[2] == t
        assert keypoint.shape[2] == t

        return video, labels, mask, keypoint
