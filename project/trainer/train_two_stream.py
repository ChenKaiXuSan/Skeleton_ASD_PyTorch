#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/trainer/train_two_stream.py
Project: /workspace/skeleton/project/trainer
Created Date: Friday June 7th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday June 7th 2024 7:50:12 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.io import write_video
from torchvision.utils import save_image, flow_to_image

from models.make_model import MakeOriginalTwoStream
from models.optical_flow import Optical_flow

from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

class TwoStreamModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type = hparams.model
        # self.fusion = hparams.fusion
        self.img_size = hparams.img_size
        self.lr = hparams.lr
        self.num_classes = hparams.model.num_classes

        # model define
        self.optical_flow_model = Optical_flow()

        self.model = MakeOriginalTwoStream(hparams)
        self.model_rgb = self.model.make_resnet(3)
        self.model_flow = self.model.make_resnet(2)

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        train steop when trainer.fit called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss
        """

        label = batch["label"].detach()  # b, c, t, h, w
        video = batch["video"].detach().float().squeeze()  # b, c, t, h, w

        label = label.repeat_interleave(video.size()[2] - 1)
        loss = self.single_logic(label, video)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        val step when trainer.fit called.

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss
            accuract: selected accuracy result.
        """

        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach()  # b

        # not use the last frame
        label = label.repeat_interleave(video.size()[2] - 1)
        loss = self.single_logic(label, video)

    def test_step(self, batch, batch_idx):
        """
        test step when trainer.test called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_
        """
         # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach()  # b

        # not use the last frame
        label = label.repeat_interleave(video.size()[2] - 1)
        loss = self.single_logic(label, video)

    def configure_optimizers(self):
        """
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        """

        optimzier = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimzier,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimzier),
                "monitor": "val_loss",
            },
        }
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type

    def single_logic(self, label: torch.Tensor, video: torch.Tensor):

        # pred the optical flow base RAFT
        # last_frame = video[:, :, -1, :].unsqueeze(dim=2) # b, c, 1, h, w
        # OF_video = torch.cat([video, last_frame], dim=2)
        video_flow = self.optical_flow_model.process_batch(video)  # b, c, t, h, w

        b, c, t, h, w = video.shape

        single_img = video[:, :, :-1, :].reshape(-1, 3, h, w)
        single_flow = video_flow.contiguous().view(-1, 2, h, w)

        # for i in range(single_img.size()[0]):
        #     save_image(single_img[i], fp='/workspace/test/rgb_%i.jpg' % i)
        #     save_image(flow_to_image(single_flow[i]).float() / 255, fp='/workspace/test/flow_%i.jpg' % i)

        # eval model, feed data here
        if self.training:
            pred_video_rgb = self.model_rgb(single_img)
            pred_video_flow = self.model_flow(single_flow)
        else:
            with torch.no_grad():
                pred_video_rgb = self.model_rgb(single_img)
                pred_video_flow = self.model_flow(single_flow)

            # squeeze(dim=-1) to keep the torch.Size([1]), not null.
            loss_rgb = F.cross_entropy(
                pred_video_rgb.squeeze(dim=-1), label.float()
            )
            loss_flow = F.cross_entropy(
                pred_video_flow.squeeze(dim=-1), label.float()
            )

            loss = loss_rgb + loss_flow

            self.save_log([pred_video_rgb, pred_video_flow], label, loss)

        return loss

    def save_log(self, pred_list: list, label: torch.Tensor, loss):

        if self.training:

            preds = pred_list[0]

            # when torch.size([1]), not squeeze.
            if preds.size()[0] != 1 or len(preds.size()) != 1:
                preds = preds.squeeze(dim=-1)
                pred_softmax = torch.softmax(preds)
            else:
                pred_softmax = torch.softmax(preds)

            # video rgb metrics
            accuracy = self._accuracy(pred_softmax, label)
            precision = self._precision(pred_softmax, label)
            recall = self._recall(pred_softmax, label)
            f1_score = self._f1_score(pred_softmax, label)
            confusion_matrix = self._confusion_matrix(pred_softmax, label)

            # log to tensorboard
            self.log_dict(
                {
                    "train/loss": loss,
                    "train/acc": accuracy,
                    "train/precision": precision,
                    "train/recall": recall,
                    "train/f1_score": f1_score,
                },
                on_epoch=True,
                on_step=True,
                batch_size=label.size()[0],
            )

        else:

            preds = pred_list[0]

            # when torch.size([1]), not squeeze.
            if preds.size()[0] != 1 or len(preds.size()) != 1:
                preds = preds.squeeze(dim=-1)
                pred_softmax = torch.sigmoid(preds)
            else:
                pred_softmax = torch.sigmoid(preds)

            # video rgb metrics
            accuracy = self._accuracy(pred_softmax, label)
            precision = self._precision(pred_softmax, label)
            recall = self._recall(pred_softmax, label)
            f1_score = self._f1_score(pred_softmax, label)
            confusion_matrix = self._confusion_matrix(pred_softmax, label)

            # log to tensorboard
            self.log_dict(
                {
                    "val/loss": loss,
                    "val/acc": accuracy,
                    "val/precision": precision,
                    "val/recall": recall,
                    "val/f1_score": f1_score,
                },
                on_epoch=True,
                on_step=True,
                batch_size=label.size()[0],
            )
