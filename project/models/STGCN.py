'''
File: STGCN.py
Project: models
Created Date: 2023-09-03 14:23:05
Author: chenkaixu
-----
Comment:
 
Have a good code time!
-----
Last Modified: 2023-09-03 14:45:44
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

'''

import torch 
import copy
from mmaction.registry import MODELS

model = dict(
type='RecognizerGCN',
backbone=dict(
    type='STGCN', graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
cls_head=dict(type='GCNHead', num_classes=2, in_channels=256))

model = MODELS.build(model)

print(model)
