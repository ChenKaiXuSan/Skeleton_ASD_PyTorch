#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/skeleton/project/train_late_fusion.py
Project: /workspace/skeleton/project
Created Date: Monday May 13th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday May 13th 2024 4:25:25 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

from typing import Any, List, Optional, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix
)

from models.make_model import MakeVideoModule

class LateFusionModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.img_size = hparams.data.img_size
        self.lr = hparams.optimizer.lr
        self.num_classes = hparams.model.model_class_num

        # define model
        self.stance_cnn = MakeVideoModule(hparams)()
        self.swing_cnn = MakeVideoModule(hparams)()

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

    def forward(self, x):
        return self.video_cnn(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):

        stance_video = batch["stance"]['video'].detach()  # b, c, t, h, w
        stance_label = batch["stance"]['label'].detach().float() # b

        swing_video = batch["swing"]['video'].detach()  # b, c, t, h, w
        swing_label = batch["swing"]['label'].detach().float() # b

        # sample_info = batch["info"] # b is the video instance number

        # keep shape 
        if stance_video.size()[0] > swing_video.size()[0]:
            stance_video = stance_video[:swing_video.size()[0]]
            stance_label = stance_label[:swing_video.size()[0]]
            label = swing_label
        elif stance_video.size()[0] < swing_video.size()[0]:
            swing_video = swing_video[:stance_video.size()[0]]
            swing_label = swing_label[:stance_video.size()[0]]
            label = stance_label
        else:
            label = stance_label

        # * slove OOM problem, cut the large batch, when >= 30 
        if stance_video.size()[0] + swing_video.size()[0] >= 30:
            stance_preds = self.stance_cnn(stance_video[:14])
            swing_preds = self.swing_cnn(swing_video[:14])
            stance_label = stance_label[:14]
            swing_label = swing_label[:14]
            label = label[:14]
        else:
            stance_preds = self.stance_cnn(stance_video)
            swing_preds = self.swing_cnn(swing_video)

        # stance loss 
        stance_loss = F.cross_entropy(stance_preds, stance_label.long())

        # swing loss 
        swing_loss = F.cross_entropy(swing_preds, swing_label.long())

        predict = (stance_preds + swing_preds) / 2 
        predict_softmax = torch.softmax(predict, dim=1)

        # loss = F.cross_entropy(predict, label.long())
        loss = (stance_loss + swing_loss) / 2

        self.log("train/loss", loss, on_epoch=True, on_step=True, batch_size=label.size()[0])

        # log metrics
        video_acc = self._accuracy(predict_softmax, label)
        video_precision = self._precision(predict_softmax, label)
        video_recall = self._recall(predict_softmax, label)
        video_f1_score = self._f1_score(predict_softmax, label)
        video_confusion_matrix = self._confusion_matrix(predict_softmax, label)
        
        self.log_dict(
            {
                "train/video_acc": video_acc,
                "train/video_precision": video_precision,
                "train/video_recall": video_recall,
                "train/video_f1_score": video_f1_score,
            }, 
            on_epoch=True, on_step=True, batch_size=label.size()[0]
        )

        return loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int):

        stance_video = batch["stance"]['video'].detach()  # b, c, t, h, w
        stance_label = batch["stance"]['label'].detach().float() # b

        swing_video = batch["swing"]['video'].detach()  # b, c, t, h, w
        swing_label = batch["swing"]['label'].detach().float() # b

        # sample_info = batch["info"] # b is the video instance number

        # keep shape 
        if stance_video.size()[0] > swing_video.size()[0]:
            stance_video = stance_video[:swing_video.size()[0]]
            stance_label = stance_label[:swing_video.size()[0]]
            label = swing_label
        elif stance_video.size()[0] < swing_video.size()[0]:
            swing_video = swing_video[:stance_video.size()[0]]
            swing_label = swing_label[:stance_video.size()[0]]
            label = stance_label
        else:
            label = stance_label

        # * slove OOM problem, cut the large batch, when >= 30 
        if stance_video.size()[0] + swing_video.size()[0] >= 30:
            stance_preds = self.stance_cnn(stance_video[:14])
            swing_preds = self.swing_cnn(swing_video[:14])
            stance_label = stance_label[:14]
            swing_label = swing_label[:14]
            label = label[:14]
        else:
            stance_preds = self.stance_cnn(stance_video)
            swing_preds = self.swing_cnn(swing_video)

        # stance loss 
        stance_loss = F.cross_entropy(stance_preds, stance_label.long())

        # swing loss 
        swing_loss = F.cross_entropy(swing_preds, swing_label.long())

        predict = (stance_preds + swing_preds) / 2 
        predict_softmax = torch.softmax(predict, dim=1)

        # loss = F.cross_entropy(predict, label.long())
        loss = (stance_loss + swing_loss) / 2

        self.log("val/loss", loss, on_epoch=True, on_step=True, batch_size=label.size()[0])

        # log metrics
        video_acc = self._accuracy(predict_softmax, label)
        video_precision = self._precision(predict_softmax, label)
        video_recall = self._recall(predict_softmax, label)
        video_f1_score = self._f1_score(predict_softmax, label)
        video_confusion_matrix = self._confusion_matrix(predict_softmax, label)
        
        self.log_dict(
            {
                "val/video_acc": video_acc,
                "val/video_precision": video_precision,
                "val/video_recall": video_recall,
                "val/video_f1_score": video_f1_score,
            }, 
            on_epoch=True, on_step=True, batch_size=label.size()[0]
        )



    def test_step(self, batch: torch.Tensor, batch_idx: int):

        stance_video = batch["stance"]['video'].detach()  # b, c, t, h, w
        stance_label = batch["stance"]['label'].detach().float() # b

        swing_video = batch["swing"]['video'].detach()  # b, c, t, h, w
        swing_label = batch["swing"]['label'].detach().float() # b

        # sample_info = batch["info"] # b is the video instance number

        # keep shape 
        if stance_video.size()[0] > swing_video.size()[0]:
            stance_video = stance_video[:swing_video.size()[0]]
            stance_label = stance_label[:swing_video.size()[0]]
            label = swing_label
        elif stance_video.size()[0] < swing_video.size()[0]:
            swing_video = swing_video[:stance_video.size()[0]]
            swing_label = swing_label[:stance_video.size()[0]]
            label = stance_label
        else:
            label = stance_label

        # * slove OOM problem, cut the large batch, when >= 30 
        if stance_video.size()[0] + swing_video.size()[0] >= 30:
            stance_preds = self.stance_cnn(stance_video[:14])
            swing_preds = self.swing_cnn(swing_video[:14])
            stance_label = stance_label[:14]
            swing_label = swing_label[:14]
            label = label[:14]
        else:
            stance_preds = self.stance_cnn(stance_video)
            swing_preds = self.swing_cnn(swing_video)

        # stance loss 
        stance_loss = F.cross_entropy(stance_preds, stance_label.long())

        # swing loss 
        swing_loss = F.cross_entropy(swing_preds, swing_label.long())

        predict = (stance_preds + swing_preds) / 2 
        predict_softmax = torch.softmax(predict, dim=1)

        # loss = F.cross_entropy(predict, label.long())
        loss = (stance_loss + swing_loss) / 2

        self.log("test/loss", loss, on_epoch=True, on_step=True, batch_size=label.size()[0])

        # log metrics
        video_acc = self._accuracy(predict_softmax, label)
        video_precision = self._precision(predict_softmax, label)
        video_recall = self._recall(predict_softmax, label)
        video_f1_score = self._f1_score(predict_softmax, label)
        video_confusion_matrix = self._confusion_matrix(predict_softmax, label)
        
        self.log_dict(
            {
                "test/video_acc": video_acc,
                "test/video_precision": video_precision,
                "test/video_recall": video_recall,
                "test/video_f1_score": video_f1_score,
            }, 
            on_epoch=True, on_step=True, batch_size=label.size()[0]
        )


    def configure_optimizers(self):
        """
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.trainer.estimated_stepping_batches, verbose=True, 
                ),
                "monitor": "train/loss",
            },
        }
