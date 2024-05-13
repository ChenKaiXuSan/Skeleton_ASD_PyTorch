#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/skeleton/project/trainer/__init__.py
Project: /workspace/skeleton/project/trainer
Created Date: Monday May 13th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday May 13th 2024 11:47:21 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import sys
sys.path.append('/workspace/skeleton/project/trainer')

from train_late_fusion import * 
from train_single import *
from train_temporal_mix import *