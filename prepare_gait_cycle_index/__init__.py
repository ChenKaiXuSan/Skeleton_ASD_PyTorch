'''
File: __init__.py
Project: prepare_gait_dataset
Created Date: 2023-10-30 06:56:12
Author: chenkaixu
-----
Comment:
 
Have a good code time!
-----
Last Modified: Sunday September 3rd 2023 1:13:52 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

'''
import os, logging, shutil, sys, math
# sys.path.append("/workspace/skeleton")

from main import * 
from prepare.prepare_gait_dataset.preprocess import * 
from yolov8 import * 
