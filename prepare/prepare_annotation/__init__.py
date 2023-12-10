'''
File: __init__.py
Project: prepare_annotation
Created Date: 2023-10-26 05:19:46
Author: chenkaixu
-----
Comment:
 
Have a good code time!
-----
Last Modified: 2023-10-26 05:39:05
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

'''
import os, logging, shutil, sys, math
sys.path.append("/workspace/skeleton")
os.environ["HYDRA_FULL_ERROR"] = "1"

from prepare_annotation.one_gait_cycle import define_gait_cycle
from prepare_annotation.videos_to_frames import * 