#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/project/analysis/draw_CAM.py
Project: /workspace/project/analysis
Created Date: Tuesday November 5th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday November 5th 2024 1:10:31 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
# %%
# load model
import sys
import os

# Append the project root directory
project_root = os.path.abspath("..")  # Adjust depending on your directory structure
project_root = os.path.abspath("/workspace/project/project")
sys.path.append(project_root)

from trainer.train_temporal_mix import TemporalMixModule

# Path to the checkpoint file
checkpoint_path = "/workspace/project/logs/paper/3D_CNN/peridicty_symmetry/temporal_mix_true_none/resnet/2024-05-15/3/14-19-39/0/version_0/checkpoints/4-0.48-0.7002.ckpt"

# Load the model from the checkpoint
model = TemporalMixModule.load_from_checkpoint(checkpoint_path, hparams_file="/workspace/project/logs/paper/3D_CNN/peridicty_symmetry/temporal_mix_true_none/resnet/2024-05-15/3/14-19-39/0/version_0/hparams.yaml")

# %%
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

with open("/workspace/project/logs/paper/3D_CNN/peridicty_symmetry/temporal_mix_true_none/resnet/2024-05-15/3/14-19-39/0/version_0/hparams.yaml", "r") as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

config = OmegaConf.create(model.hparams["hparams"])

# %%
model.hparams

# %%
from dataloader.data_loader import WalkDataModule
import json

path = "/workspace/data/seg_gait_index_mapping/3/none/index.json"
with open(path, "r") as f:
    fold_dataset_idx = json.load(f)

fold_dataset_idx = {
    "0": [
        [
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0001.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0002.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0003.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0004.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0005.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0006.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0007.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0001.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0002.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0003.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0004.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0005.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0006.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0007.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0001.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0002.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0003.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0004.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0005.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0006.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0007.json",],
        [
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0003.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0001.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0002.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0003.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0004.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0005.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0006.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0007.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0001.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0002.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0003.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0004.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0005.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0006.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0007.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0001.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0002.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0003.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0004.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0005.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0006.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0007.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0001.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0002.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0003.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0004.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0005.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0006.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160120_ASD_lat__V1-0007.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0001.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0002.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0003.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0004.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0005.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0006.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160426_ASD_lat__V1-0007.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0001.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0002.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0003.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0004.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0005.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0006.json",
            "/workspace/data/segmentation_dataset_512/json_mix/ASD/20160523_1_ASD_lat__V1-0007.json",
        ],
    ],  
}
data_module = WalkDataModule(config, fold_dataset_idx['0'])

# %%
import torch 
from pathlib import Path

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    FullGrad,
    GradCAMPlusPlus,
    AblationCAM,
    ScoreCAM,
    LayerCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import visualization as viz

def save_CAM(
    config,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    inp_label,
    fold,
    flag,
    i,
    random_index,
):
    # guided grad cam method
    target_layer = [model.blocks[-2].res_blocks[-1]]
    # target_layer = [model.model.blocks[-2]]

    cam = GradCAMPlusPlus(model, target_layer)

    # save the CAM
    save_path = Path(config.train.log_path) / "CAM" / f"fold{fold}" / flag

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    for idx, num in enumerate(random_index):

        grayscale_cam = cam(
            input_tensor[num : num + 1], aug_smooth=True, eigen_smooth=True
        )
        output = cam.outputs

        # prepare save figure
        inp_tensor = (
            input_tensor[num].permute(1, 2, 3, 0)[-1].cpu().detach().numpy()
        )  # display original image
        cam_map = grayscale_cam.squeeze().mean(axis=2, keepdims=True)

        figure, axis = viz.visualize_image_attr(
            cam_map,
            inp_tensor,
            method="blended_heat_map",
            sign="positive",
            show_colorbar=True,
            cmap="jet",
            title=f"label: {int(inp_label[num])}, pred: {output.argmax(dim=1)[0]}",
        )

        figure.savefig(
            save_path / f"fold{fold}_{flag}_step{i}_num{idx}_{num}.png",
        )


# %%
import matplotlib.pyplot as plt

data_module.setup()
test_dataloader = data_module.test_dataloader()

batch = next(iter(test_dataloader))

# input and label
video = (
    batch["video"].detach().to(f"cuda:{config.train.gpu_num}")
)  # b, c, t, h, w
label = (
    batch["label"].detach().to(f"cuda:{config.train.gpu_num}")
)  # b, class_num    

model.eval().to(f"cuda:{config.train.gpu_num}")

# pred the video frames
with torch.no_grad():
    preds = model(video)

# when torch.size([1]), not squeeze.
if preds.size()[0] != 1 or len(preds.size()) != 1:
    preds = preds.squeeze(dim=-1)
    preds_softmax = torch.softmax(preds, dim=1)
else:
    preds_softmax = torch.softmax(preds, dim=1)

# guided grad cam method
target_layer = [model.video_cnn.blocks[-2].res_blocks[-1]]
# target_layer = [model.model.blocks[-2]]

cam = GradCAMPlusPlus(model, target_layer)

# save the CAM
save_path = Path('.') / "CAM"

if save_path.exists() is False:
    save_path.mkdir(parents=True)

grayscale_cam = cam(
    video[0:1], aug_smooth=False, eigen_smooth=True
)
output = cam.outputs

# prepare save figure
inp_tensor = (
    video[0].permute(1, 2, 3, 0)[-1].cpu().detach().numpy()
)  # display original image
cam_map = grayscale_cam.squeeze().mean(axis=2, keepdims=True)

figure, axis = viz.visualize_image_attr(
    cam_map,
    inp_tensor,
    method="blended_heat_map",
    sign="positive",
    show_colorbar=True,
    cmap="jet",
    # title=f"label: {int(inp_label[num])}, pred: {output.argmax(dim=1)[0]}",
)

# figure.savefig(
#     save_path / f"fold{fold}_{flag}_step{i}_num{idx}_{num}.png",
# )

plt.imshow(figure)


