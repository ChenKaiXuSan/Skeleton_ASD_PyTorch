#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/misc/result_analysis.py
Project: /workspace/skeleton/misc
Created Date: Monday December 18th 2023
Author: Kaixu Chen
-----
Comment:
This script is used to analyze the results of the trained model, 
including the accuracy, precision, recall, f1 score, auroc, and confusion matrix.
And save the results into the .pt file, with pred and label.

Have a good code time :)
-----
Last Modified: Thursday April 4th 2024 3:56:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2023 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

05-04-2024	Kaixu Chen	now add this function into main.py file. So not need this file for more.
"""

import os, hydra, warnings

warnings.filterwarnings("ignore")

import logging, time, sys, json, yaml, csv, shutil, copy

sys.path.append("/workspace/skeleton/")
sys.path.append("/workspace/skeleton/project")
from pathlib import Path
import torch

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
)

from project.dataloader.data_loader import WalkDataModule
from project.train import GaitCycleLightningModule

from pytorch_lightning import seed_everything

from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.io.image import write_png


def get_inference(test_data, model):

    total_pred_list = []
    total_label_list = []

    for i, batch in enumerate(test_data):

        pred_list = []
        label_list = []

        # input and label
        video = batch["video"].detach().to("cuda:0")  # b, c, t, h, w

        label = batch["label"].detach().to("cuda:0")  # b, class_num

        model.eval()

        # write_png((video[0, :, 2, ...]*255).cpu().to(torch.uint8), filename=f'test{i}.png')

        # pred the video frames
        with torch.no_grad():
            preds = model(video)

        # when torch.size([1]), not squeeze.
        if preds.size()[0] != 1 or len(preds.size()) != 1:
            preds = preds.squeeze(dim=-1)
            preds_softmax = torch.softmax(preds, dim=1)
        else:
            preds_softmax = torch.softmax(preds, dim=1)

        pred_list.append(preds_softmax.tolist())
        label_list.append(label.tolist())

        for i in pred_list:
            for number in i:
                total_pred_list.append(number)

        for i in label_list:
            for number in i:
                total_label_list.append(number)

    pred = torch.tensor(total_pred_list)
    label = torch.tensor(total_label_list)

    return pred, label


def get_visualization(test_data, model):

    target_layer = [model.video_cnn.blocks[-2].res_blocks[-1]]

    cam = LayerCAM(model, target_layer)

    targets = [ClassifierOutputTarget(-1)]

    batch = next(iter(test_data))
    video = batch["video"].detach().to("cuda:0")  # b, c, t, h, w

    grayscale_cam = cam(
        input_tensor=video[0].unsqueeze(dim=0), aug_smooth=True, eigen_smooth=True
    )
    print(grayscale_cam.shape)

def main():

    sampler = "none"

    val_dataset_idx = "/workspace/data/seg_gait_index_mapping/2/none/index.json"

    # * this is best ckpt trained by the model
    ckpt_path = "/workspace/skeleton/logs/resnet/2024-04-06/2/06-50-05/0/version_0/checkpoints/0-0.67-0.5916.ckpt"

    hparams = torch.load(ckpt_path, map_location='cpu')['hyper_parameters']['hparams']

    fold = ckpt_path.split("/")[-4]

    # hparams.model.model = ckpt_path.split("/")[4]
    hparams.train.gpu_num = 0
    # hparams.train.temporal_mix=True
    # hparams.train.batch_size = 64

    # prepare dataset mapping     

    fold_dataset_idx = json.load(open(val_dataset_idx, "r"))

    # unpack the dataset mapping
    for k, v in fold_dataset_idx.items():
        # train mapping, include the gait cycle index
        train_mapping_idx = v[0]
        fold_dataset_idx[k][0] = [Path(i) for i in train_mapping_idx]

        # val mapping, include the gait cycle index 
        val_mapping_idx = v[1]
        fold_dataset_idx[k][1] = [Path(i) for i in val_mapping_idx]

        # train video path
        train_video_idx = v[2]
        fold_dataset_idx[k][2] = Path(train_video_idx)

        # val video path
        val_dataset_idx = v[3]
        fold_dataset_idx[k][3] = Path(val_dataset_idx)

    seed_everything(42, workers=True)

    # load model
    classification_module = (
        GaitCycleLightningModule(hparams)
        .load_from_checkpoint(ckpt_path)
        .to(f"cuda:{hparams.train.gpu_num}")
    )

    # load dataset
    data_module = WalkDataModule(hparams, fold_dataset_idx[fold])
    data_module.setup()

    # define metrics
    _accuracy = MulticlassAccuracy(hparams.model.model_class_num)
    _precision = MulticlassPrecision(hparams.model.model_class_num)
    _recall = MulticlassRecall(hparams.model.model_class_num)
    _f1_score = MulticlassF1Score(hparams.model.model_class_num)
    _auroc = MulticlassAUROC(hparams.model.model_class_num)
    _confusion_matrix = MulticlassConfusionMatrix(hparams.model.model_class_num)

    # get_visualization(data_module.test_dataloader(), classification_module)

    pred, label = get_inference(data_module.test_dataloader(), classification_module)

    # save the results
    torch.save(
        pred,
        Path("/workspace/skeleton/analysis")
        / f"{hparams.model.model}_{sampler}_{fold}_pred.pt",
    )
    torch.save(
        label,
        Path("/workspace/skeleton/analysis")
        / f"{hparams.model.model}_{sampler}_{fold}_label.pt",
    )

    logging.info(
        f"save the pred and label into {Path('/workspace/skeleton/analysis')} / {hparams.model.model}_{sampler}_{fold}_pred.pt and {hparams.model.model}_{sampler}_{fold}_label.pt"
    )

    print("*" * 100)
    print("accuracy: %s" % _accuracy(pred, label))
    print("precision: %s" % _precision(pred, label))
    print("_binary_recall: %s" % _recall(pred, label))
    print("_binary_f1: %s" % _f1_score(pred, label))
    print("_aurroc: %s" % _auroc(pred, label))
    print("_confusion_matrix: %s" % _confusion_matrix(pred, label))
    print("#" * 100)


if __name__ == "__main__":

    config = main()
