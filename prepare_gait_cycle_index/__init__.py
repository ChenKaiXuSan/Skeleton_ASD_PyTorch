'''
File: __init__.py
Project: prepare_gait_dataset
Created Date: 2023-10-30 06:56:12
Author: chenkaixu
-----
Comment:
 
Have a good code time!
-----
Last Modified: Monday April 8th 2024 2:50:12 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

'''
import sys, os

curr_path = os.path.dirname(__file__)
sys.path.append(curr_path)

# from make_model import *
from main import * 
from preprocess import * 
from yolov8 import * 
