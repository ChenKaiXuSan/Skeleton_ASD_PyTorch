#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/skeleton/prepare_video/yolov8_SORT.py
Project: /workspace/skeleton/prepare_video
Created Date: Saturday November 25th 2023
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday November 25th 2023 1:17:36 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2023 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

# %%
import torch
from torchvision.transforms.functional import crop, pad, resize
from torchvision.utils import save_image
from torchvision.io import read_image, read_video, write_video, write_png

import os, shutil, sys, multiprocessing
from argparse import ArgumentParser

import matplotlib.pyplot as plt 

from ultralytics import YOLO


# %%

def clip_pad_with_bbox(imgs: torch.tensor, boxes: list, img_size: int = 256, bias:int = 10):
        '''
        based torchvision function to crop, pad, resize img.

        clip with the bbox, (x1-bias, y1) and padd with the (gap-bais) in left and right.

        Args:
            imgs (list): imgs with (h, w, c)
            boxes (list): (x1, y1, x2, y2)
            img_size (int, optional): croped img size. Defaults to 256.
            bias (int, optional): the bias of bbox, with the (x1-bias) and (x2+bias). Defaults to 5.

        Returns:
            tensor: (c, t, h, w)
        ''' 
        object_list = []
        
        for box in boxes:

            x1, y1, x2, y2 = map(int, box) # dtype must int for resize, crop function

            box_width = x2 - x1
            box_height = y2 - y1 

            width_gap = int(((box_height - box_width) / 2)) # keep int type

            img = imgs # (h, w, c) to (c, h, w), for pytorch function

            # give a bias for the left and right crop bbox.
            croped_img = crop(img, top=y1, left=(x1 - bias), height=box_height, width=(box_width + 2 * bias))

            pad_img = pad(croped_img, padding=(width_gap - bias, 0), fill=0)

            resized_img = resize(pad_img, size=(img_size, img_size))

            object_list.append(resized_img)

        return object_list # c, t, h, w

# %%

def del_folder(path, *args):
    '''
    delete the folder which path/version

    Args:
        path (str): path
        version (str): version
    '''
    if os.path.exists(os.path.join(path, *args)):
        shutil.rmtree(os.path.join(path, *args))


def make_folder(path, *args):
    '''
    make folder which path/version

    Args:
        path (str): path
        version (str): version
    '''
    if not os.path.exists(os.path.join(path, *args)):
        os.makedirs(os.path.join(path, *args))
        print("success make dir! where: %s " % os.path.join(path, *args))
    # else:
        # print("The target path already exists! where: %s " % os.path.join(path, *args))


def main(parames:ArgumentParser, disease:str):

    # DISEASE = ['ASD', 'DHS', 'LCS', 'HipOA', 'normal']

    DATA_PATH = parames.data_path
    SAVE_PATH = parames.save_path
    IMG_SIZE = parames.img_size 

    # Load a model
    model = YOLO(parames.ckpt_path)  # load an official segmentation model


    file_list = os.listdir(os.path.join(DATA_PATH, disease))
    sorted(file_list)
    
    for file in file_list:
        try:
            one_file = os.listdir(os.path.join(DATA_PATH, disease, file))
            for one in one_file:
                # make generator
                results = model.track(source=os.path.join(DATA_PATH, disease, file, one), conf=0.8, iou=0.9, save_crop=False, classes=0, vid_stride=True, stream=True, project='project', name='name')

                save_path = os.path.join(SAVE_PATH, disease, file)

                one_file_name = one.split('.')[0]
                # make_folder(os.path.join(save_path, disease))
                make_folder(save_path)

                # output log to file
                log_path = os.path.join(save_path, '%s.log' % one_file_name)
                sys.stdout = open(log_path, 'w')


                for i, r in enumerate(results):
                    
                    orig_img = r.orig_img
                    img_id = r.boxes.id
                    boxes = r.boxes

                    if len(boxes.data.tolist()) != 0 and img_id != None:

                        resized_img = clip_pad_with_bbox(torch.tensor(orig_img[:,:,[2,1,0]]).permute(2,0,1), boxes=boxes.xyxy.tolist(), img_size=IMG_SIZE, bias=0)

                        for num, id in enumerate(img_id.tolist()):
                            make_folder(os.path.join(save_path, one_file_name, str(int(id))))
                            # save_image(tensor=resized_img[int(num)].float()/255, fp=os.path.join(save_path, str(int(id)), '%d.png' %i))
                            write_png(input=resized_img[int(num)], filename=os.path.join(save_path, one_file_name, str(int(id)), '%d.png' % i))
                    else:
                        print('drop %s, %d frame' % (os.path.join(DATA_PATH, disease, file, one), i))
        except:
            continue

# %%
def get_parameters():
    """
    get_parameters from python script.

    Returns:
        parser: include parsers.
    """
    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--img_size', type=int, default=512)

    # Path
    parser.add_argument('--data_path', type=str, default="/workspace/data/raw_data", help='meta dataset path')
    parser.add_argument('--save_path', type=str, default="/workspace/data/four_view_data_512",
                        help="split dataset with detection method.")
    parser.add_argument('--ckpt_path', type=str, default="/workspace/skeleton/ckpt/yolov8x-seg.pt",)

    return parser.parse_known_args()

# %% 
if __name__ == '__main__':
    
    parames, unknown = get_parameters()

    # define process
    normal_process = multiprocessing.Process(target=main, args=(parames, 'normal'))
    # ASD_process = multiprocessing.Process(target=main, args=(parames, 'ASD'))
    # DHS_process = multiprocessing.Process(target=main, args=(parames, 'DHS'))
    # LCS_process = multiprocessing.Process(target=main, args=(parames, 'LCS'))
    # HipOA_process = multiprocessing.Process(target=main, args=(parames, 'HipOA'))

    # start process 
    # ASD_process.start()
    # DHS_process.start()
    # LCS_process.start()
    # HipOA_process.start()

    normal_process.start()

    # main(parames, 'normal')
    

    
# # %%
# img = next(results)
# img.boxes.boxes == 0

# # %% 
# img.boxes.data.tolist()

# # %%
# import matplotlib.pyplot as plt 

# plt.imshow(img.orig_img[:,:,[2,1,0]]), img.boxes.id

# # %%
# plt.imshow(resized_img[1].permute(1,2,0))

# # %%

# x1,y1,x2,y2 = map(int, img.boxes.xyxy.tolist()[0])
# crop_img = crop(torch.tensor(img.orig_img[:,:,[2,1,0]]).permute(2,0,1), top=y1, left=x1, height=abs(y2-y1), width=abs(x2-x1))
# plt.imshow(crop_img.permute(1,2,0))

# # %%
# write_video(filename='text.mp4', video_array=torch.stack(lis, dim=0), fps=30)

# %%
