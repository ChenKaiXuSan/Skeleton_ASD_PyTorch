#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/skeleton/project/models/mm_model.py
Project: /workspace/skeleton/project/models
Created Date: Wednesday June 12th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday June 12th 2024 1:27:27 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

from mmaction.apis import init_recognizer
from mmaction.registry import MODELS

import torch

ckpt_file = "/workspace/skeleton/ckpt/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221222-4c0ed77e.pth"

model = init_recognizer("/workspace/skeleton/project/models/agcn.py", ckpt_file, device='cuda:0')

print(model)

# N, M, T, V, C = 1, 1, 2, 100, 18, 2
inp = torch.randn(1, 2, 2, 8, 17, 3).cuda()
print(inp.shape)
out = model(inp)
print(out.shape)