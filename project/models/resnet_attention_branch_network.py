#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/project/project/models/resnet_attention_branch_network.py
Project: /workspace/project/project/models
Created Date: Tuesday November 26th 2024
Author: Kaixu Chen
-----
Comment:

This is a compare experiment for the ResNet-3D CNN model with attention branch network.

Have a good code time :)
-----
Last Modified: Tuesday November 26th 2024 1:28:51 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchvideo.models import x3d, resnet, slowfast
from torchvision.models import resnet50, mobilenet_v3_large, efficientnet_v2_l, alexnet

class MakeVideoModule(nn.Module):
    '''
    the module zoo from the PytorchVideo lib, to make the different 3D model.

    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_name = hparams.model.model
        self.model_class_num = hparams.model.model_class_num
        self.model_depth = hparams.model.model_depth
        self.transfer_learning = hparams.train.transfer_learning

    def make_walk_resnet(self, input_channel:int = 3) -> nn.Module:

        if self.transfer_learning:
            slow = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
            
            # for the folw model and rgb model 
            slow.blocks[0].conv = nn.Conv3d(input_channel, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
            # change the knetics-400 output 400 to model class num
            slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

        else:
            slow = resnet.create_resnet(
                input_channel=input_channel,
                model_depth=self.model_depth,
                model_num_class=self.model_class_num,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )

        return slow

    def make_walk_x3d(self, input_channel:int = 3) -> nn.Module:

        if self.transfer_learning:
            # x3d l model, param 6.15 with 16 frames. more smaller maybe more faster.
            # top1 acc is 77.44
            model = torch.hub.load("facebookresearch/pytorchvideo:main", model='x3d_m', pretrained=True)
            model.blocks[0].conv.conv_t = nn.Conv3d(
                in_channels=input_channel, 
                out_channels=24,
                kernel_size=(1,3,3),
                stride=(1,2,2),
                padding=(0,1,1),
                bias=False
                )
            
            model.blocks[-1].proj = nn.Linear(2048, self.model_class_num)
            model.blocks[-1].activation = None

        else:
            model = x3d.create_x3d(
                input_channel=3,
                input_clip_length=16,
                input_crop_size=224,
                model_num_class=1,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
                head_activation=None,
            )

        return model
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:

        if self.model_name == "resnet":
            return self.make_walk_resnet()
        elif self.model_name == "x3d":
            return self.make_walk_x3d()
        else:
            raise KeyError(f"the model name {self.model_name} is not in the model zoo")
