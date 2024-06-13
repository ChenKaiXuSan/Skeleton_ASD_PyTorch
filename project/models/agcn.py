#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/skeleton/project/models/agcn.py
Project: /workspace/skeleton/project/models
Created Date: Wednesday June 12th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday June 12th 2024 3:08:24 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='AAGCN',
        graph_cfg=dict(layout='coco', mode='spatial'),
        gcn_attention=False),  # degenerate AAGCN to AGCN
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))
