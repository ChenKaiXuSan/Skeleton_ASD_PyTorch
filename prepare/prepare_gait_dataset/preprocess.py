'''
File: preprocess.py
Project: models
Created Date: 2023-09-03 13:44:00
Author: chenkaixu
-----
Comment:
preprocess script include the pose estimation, optical flow, and mask.
This is used for define the one gait cycle from video.
 
Have a good code time!
-----
Last Modified: Monday October 30th 2023 6:50:52 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

'''


import shutil, logging
from pathlib import Path
from typing import Any

import torch  
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.io import write_png
from torchvision.utils import flow_to_image

# from optical_flow import OpticalFlow
from prepare.prepare_gait_dataset.yolov8 import  MultiPreprocess

class Preprocess(nn.Module):

    def __init__(self, config) -> None:
        super(Preprocess, self).__init__()
        # self.of_model = OpticalFlow()
        self.yolo_model = MultiPreprocess(config.YOLO)
    
    def save_img(self, batch:torch.tensor, flag:str, epoch_idx:int):
        """
        save_img save batch to png, for analysis.

        Args:
            batch (torch.tensor): the batch of imgs (b, c, t, h, w)
            flag (str): flag for filename, ['optical_flow', 'mask']
            batch_idx (int): epoch index.
        """        
        
        pref_Path = Path('/workspace/skeleton/logs/img_test')
        # only rmtree in 1 epoch, and in one flag (mask).
        if pref_Path.exists() and epoch_idx == 0 and flag == 'mask':
            shutil.rmtree(pref_Path)
            pref_Path.mkdir(parents=True)
            
        b, c, t, h, w = batch.shape

        for b_idx in range(b):
            for frame in range(t):
                
                if flag == 'optical_flow':
                    inp = flow_to_image(batch[b_idx, :, frame, ...])
                else:
                    inp = batch[b_idx, :, frame, ...] * 255
                    inp = inp.to(torch.uint8)
                    
                write_png(input=inp.cpu(), filename= str(pref_Path.joinpath(f'{flag}_{epoch_idx}_{b_idx}_{frame}.png')))
        
        logging.info('='*20)
        logging.info('finish save %s' % flag)

    def shape_check(self, check: list):
        """
        shape_check check the given value shape, and assert the shape.

        check list include:
        # batch, (b, c, t, h, w)
        # label, (b)
        # bbox, (b, t, 4) (cxcywh)
        # mask, (b, 1, t, h, w)
        # keypoint, (b, t, 17, 2)

        Args:
            check (list): checked value, in list.
        """        

        # first value in list is video, use this as reference.
        b, c, t, h, w = check[0].shape

        # frame check, we just need start from 1.
        for ck in check[0:]:
            
            # for label shape
            if len(ck.shape) == 1: assert ck.shape[0] == b 
            # for bbox shape
            elif len(ck.shape) == 3: assert ck.shape[0] == b and ck.shape[1] == t
            # for mask shape
            elif len(ck.shape) == 5: assert ck.shape[0] == b and ck.shape[2] == t
            # for keypoint shape
            elif len(ck.shape) == 4: assert ck.shape[0] == b and ck.shape[1] == t and ck.shape[2] == 17

    def forward(self, batch: torch.tensor, labels: list, batch_idx: int):
        """
        forward preprocess method for one batch, use yolo and RAFT.

        Args:
            batch (torch.tensor): batch imgs, (b, c, t, h, w)
            labels (torch.tensor): batch labels, (b) # not use.
            batch_idx (int): epoch index.

        Returns:
            list: list for different moddailty, return optical flow, mask, and pose keypoints.
        """
                
        b, c, t, h, w = batch.shape

        # process mask, pose
        video, labels, bbox, mask, pose = self.yolo_model(batch, labels)

        # shape check
        self.shape_check([video, labels, mask, bbox, pose])

        # self.save_img(mask, flag='mask', epoch_idx=batch_idx)

        # process optical flow
        # optical_flow = self.of_model(video)
        
        # shape check
        # self.shape_check([video, optical_flow])

        # self.save_img(optical_flow, flag='optical_flow', epoch_idx=batch_idx)
        
        # return video, labels, optical_flow, bbox, mask, pose
        return video, labels, None, bbox, mask, pose
