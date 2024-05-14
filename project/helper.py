#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/helper.py
Project: /workspace/skeleton/project
Created Date: Tuesday May 14th 2024
Author: Kaixu Chen
-----
Comment:
This is a helper script to save the results of the training.
The saved items include:
1. the prediction and label for the further analysis.
2. the metrics for the model evaluation.
3. the confusion matrix for the model evaluation.
4. the class activation map for the model evaluation.

This script is executed at the end of each training in main.py file.

Have a good code time :)
-----
Last Modified: Tuesday May 14th 2024 3:23:52 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os, logging
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
)


def save_helper(config, model, dataloader, fold):

    if config.train.experiment == "late_fusion":
        total_pred, total_label = save_inference_late_fusion(
            config, model, dataloader, fold
        )
    else:
        total_pred, total_label = save_inference(config, model, dataloader, fold)

    save_metrics(total_pred, total_label, fold, config)
    save_CM(total_pred, total_label, fold, config)


def save_inference_late_fusion(config, model, dataloader, fold):

    total_pred_list = []
    total_label_list = []

    device = f"cuda:{config.train.gpu_num}"

    test_dataloader = dataloader.test_dataloader()
    stance_cnn = model.stance_cnn
    swing_cnn = model.swing_cnn

    for i, batch in enumerate(test_dataloader):

        # if i == 50: break; # debug only

        stance_video = batch["stance"]["video"].detach().to(device)  # b, c, t, h, w
        stance_label = batch["stance"]["label"].detach().float().to(device)  # b

        swing_video = batch["swing"]["video"].detach().to(device)  # b, c, t, h, w
        swing_label = batch["swing"]["label"].detach().float().to(device)  # b

        # sample_info = batch["info"] # b is the video instance number

        # keep shape
        if stance_video.size()[0] > swing_video.size()[0]:
            stance_video = stance_video[: swing_video.size()[0]]
            label = swing_label
        elif stance_video.size()[0] < swing_video.size()[0]:
            swing_video = swing_video[: stance_video.size()[0]]
            label = stance_label
        else:
            label = stance_label

        stance_cnn.eval().to(device)
        swing_cnn.eval().to(device)

        with torch.no_grad():
            stance_preds = stance_cnn(stance_video)
            swing_preds = swing_cnn(swing_video)

        predict = (stance_preds + swing_preds) / 2
        preds_softmax = torch.softmax(predict, dim=1)

        for i in preds_softmax.tolist():
            total_pred_list.append(i)
        for i in label.tolist():
            total_label_list.append(i)

    pred = torch.tensor(total_pred_list)
    label = torch.tensor(total_label_list)

    # save the results
    save_path = Path(config.train.log_path) / "best_preds"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    torch.save(
        pred,
        save_path / f"{config.model.model}_{config.data.sampling}_{fold}_pred.pt",
    )
    torch.save(
        label,
        save_path / f"{config.model.model}_{config.data.sampling}_{fold}_label.pt",
    )

    logging.info(
        f"save the pred and label into {save_path} / {config.model.model}_{config.data.sampling}_{fold}"
    )

    return pred, label


def save_inference(config, model, dataloader, fold):

    total_pred_list = []
    total_label_list = []

    test_dataloader = dataloader.test_dataloader()

    for i, batch in enumerate(test_dataloader):

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

        for i in preds_softmax.tolist():
            total_pred_list.append(i)
        for i in label.tolist():
            total_label_list.append(i)

    pred = torch.tensor(total_pred_list)
    label = torch.tensor(total_label_list)

    # save the results
    save_path = Path(config.train.log_path) / "best_preds"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    torch.save(
        pred,
        save_path / f"{config.model.model}_{config.data.sampling}_{fold}_pred.pt",
    )
    torch.save(
        label,
        save_path / f"{config.model.model}_{config.data.sampling}_{fold}_label.pt",
    )

    logging.info(
        f"save the pred and label into {save_path} / {config.model.model}_{config.data.sampling}_{fold}"
    )

    return pred, label


def save_metrics(all_pred, all_label, fold, config):

    save_path = Path(config.train.log_path) / "metrics.txt"

    # define metrics
    num_class = torch.unique(all_label).size(0)
    _accuracy = MulticlassAccuracy(num_class)
    _precision = MulticlassPrecision(num_class)
    _recall = MulticlassRecall(num_class)
    _f1_score = MulticlassF1Score(num_class)
    _auroc = MulticlassAUROC(num_class)
    _confusion_matrix = MulticlassConfusionMatrix(num_class, normalize="true")

    print("*" * 100)
    print("accuracy: %s" % _accuracy(all_pred, all_label))
    print("precision: %s" % _precision(all_pred, all_label))
    print("recall: %s" % _recall(all_pred, all_label))
    print("f1_score: %s" % _f1_score(all_pred, all_label))
    print("aurroc: %s" % _auroc(all_pred, all_label.long()))
    print("confusion_matrix: %s" % _confusion_matrix(all_pred, all_label))
    print("#" * 100)

    with open(save_path, "w") as f:
        f.writelines(f"Fold {fold}\n")
        f.writelines(f"accuracy: {_accuracy(all_pred, all_label)}\n")
        f.writelines(f"precision: {_precision(all_pred, all_label)}\n")
        f.writelines(f"recall: {_recall(all_pred, all_label)}\n")
        f.writelines(f"f1_score: {_f1_score(all_pred, all_label)}\n")
        f.writelines(f"aurroc: {_auroc(all_pred, all_label.long())}\n")
        f.writelines(f"confusion_matrix: {_confusion_matrix(all_pred, all_label)}\n")
        f.writelines("#" * 100)


def save_CM(all_pred, all_label, fold, config):

    save_path = Path(config.train.log_path) / "CM"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    # define metrics
    num_class = torch.unique(all_label).size(0)
    _confusion_matrix = MulticlassConfusionMatrix(num_class, normalize="true")

    print("_confusion_matrix: %s" % _confusion_matrix(all_pred, all_label))

    # 设置字体和标题样式
    plt.rcParams.update({"font.size": 30, "font.family": "sans-serif"})

    # 假设的混淆矩阵数据
    confusion_matrix_data = _confusion_matrix(all_pred, all_label).cpu().numpy() * 100

    axis_labels = ["ASD", "DHS", "LCS_HipOA"]

    # 使用matplotlib和seaborn绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_data,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        vmin=0,
        vmax=100,
    )
    plt.title(f"Fold {fold} (%)", fontsize=30)
    plt.ylabel("Actual Label", fontsize=30)
    plt.xlabel("Predicted Label", fontsize=30)

    plt.savefig(
        save_path / f"fold{fold}_confusion_matrix.png", dpi=300, bbox_inches="tight"
    )

    print(
        f"save the confusion matrix into {save_path} / fold{fold}_confusion_matrix.png"
    )


def save_CAM(config, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
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

    # guided grad cam method
    target_layer = [model.model.blocks[-2].res_blocks[-1]]
    # target_layer = [model.model.blocks[-2]]

    cam = LayerCAM(model, target_layer, use_cuda=True)

    targets = [ClassifierOutputTarget(-1)]

    # grayscale_cam = cam(input_tensor, aug_smooth=True, eigen_smooth=True)
