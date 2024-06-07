#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
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
'''

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
    MulticlassConfusionMatrix
)

from pytorchvideo.transforms.functional import uniform_temporal_subsample

class TwoStreamModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type = hparams.model
        self.fusion = hparams.fusion
        self.img_size = hparams.img_size
        self.lr = hparams.lr

        # model define
        self.optical_flow_model = Optical_flow()

        self.model = MakeOriginalTwoStream(hparams)
        self.model_rgb = self.model.make_resnet(3)
        self.model_flow = self.model.make_resnet(2)

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''
        train steop when trainer.fit called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss
        '''

        label = batch['label'].detach()  # b, c, t, h, w
        video = batch['video'].detach().float().squeeze()  # b, c, t, h, w

        label = label.repeat_interleave(video.size()[2] - 1)
        loss = self.single_logic(label, video)
        
        return loss

    def validation_step(self, batch, batch_idx):
        '''
    val step when trainer.fit called.

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss 
            accuract: selected accuracy result.
        '''

        # input and model define
        video = batch['video'].detach()  # b, c, t, h, w
        label = batch['label'].detach()  # b

        if self.model_type == 'multi':
            loss = self.multi_logic(label, video)
        elif self.model_type == 'single':
            # not use the last frame
            label = label.repeat_interleave(video.size()[2] - 1)
            loss = self.single_logic(label, video)
        elif self.model_type == 'multi_single':
            loss = self.multi_single_logic(label, video)
        
        return loss

    def test_step(self, batch, batch_idx):
        '''
        test step when trainer.test called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_
        '''
        pass

    def configure_optimizers(self):
        '''
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        '''

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

        if self.fusion == 'different_loss':

            # squeeze(dim=-1) to keep the torch.Size([1]), not null.
            loss_rgb = F.binary_cross_entropy_with_logits(pred_video_rgb.squeeze(dim=-1), label.float())
            loss_flow = F.binary_cross_entropy_with_logits(pred_video_flow.squeeze(dim=-1), label.float())

            loss = loss_rgb + loss_flow

            self.save_log([pred_video_rgb, pred_video_flow], label, loss)

        elif self.fusion == 'sum_loss':

            pred_sum = (pred_video_rgb + pred_video_flow).squeeze(dim=-1) / 2

            loss = F.binary_cross_entropy_with_logits(pred_sum, label.float())

            self.save_log([pred_video_rgb, pred_video_flow], label, loss)

        return loss

    def multi_logic(self, label: torch.Tensor, video: torch.Tensor):

        # pred the optical flow base RAFT
        # last_frame = video[:, :, -1, :].unsqueeze(dim=2) # b, c, 1, h, w
        # OF_video = torch.cat([video, last_frame], dim=2)
        video_flow = self.optical_flow_model.process_batch(video) # b, c, t, h, w

        # save img
        for b in range(video.size()[0]):

            for t in range(video.size()[2]-1):
                save_image(video[b,:,t,:], fp='/workspace/test/rgb_%s_%s.jpg' % (b, t))
                save_image(flow_to_image(video_flow[b,:,t,:]).float() / 255, fp='/workspace/test/flow_%s_%s.jpg' % (b, t))

        # eval model, feed data here
        if self.training:
            pred_video_rgb = self.model_rgb(video)
            pred_video_flow = self.model_flow(video_flow)
        else:
            with torch.no_grad():
                pred_video_rgb = self.model_rgb(video)
                pred_video_flow = self.model_flow(video_flow)

        if self.fusion == 'different_loss':

            # squeeze(dim=-1) to keep the torch.Size([1]), not null.
            loss_rgb = F.binary_cross_entropy_with_logits(pred_video_rgb.squeeze(dim=-1), label.float())
            loss_flow = F.binary_cross_entropy_with_logits(pred_video_flow.squeeze(dim=-1), label.float())

            loss = loss_rgb + loss_flow

            self.save_log([pred_video_rgb, pred_video_flow], label, loss)

        elif self.fusion == 'sum_loss':
            # FIXME
            pred_sum = (pred_video_rgb + pred_video_flow).squeeze(dim=-1) / 2

            loss = F.binary_cross_entropy_with_logits(pred_sum, label.float())

            self.save_log([pred_sum], label, loss)

        return loss
    
    def multi_single_logic(self, label: torch.Tensor, video: torch.Tensor):
        
        single_label = label.repeat_interleave(video.size()[2] - 1)

        # pred the optical flow base RAFT
        video_flow = self.optical_flow_model.process_batch(video) # b, c, t, h, w
        # extract 16 for 3D CNN
        video = uniform_temporal_subsample(video[:, :, :-1, :].cpu(), 16).cuda()

        # save img
        # for b in range(video.size()[0]):

        #     for t in range(video.size()[2]-1):
        #         save_image(video[b,:,t,:], fp='/workspace/test/rgb_%s_%s.jpg' % (b, t))
        #         save_image(flow_to_image(video_flow[b,:,t,:]).float() / 255, fp='/workspace/test/flow_%s_%s.jpg' % (b, t))

        b, c, t, h, w = video.shape

        single_flow = video_flow.contiguous().view(-1, 2, h, w)

        # eval model, feed data here
        if self.training:
            pred_video_rgb = self.model_rgb(video)
            pred_video_flow = self.model_flow(single_flow)
        else:
            with torch.no_grad():
                pred_video_rgb = self.model_rgb(video)
                pred_video_flow = self.model_flow(single_flow)

        if self.fusion == 'different_loss':

            # squeeze(dim=-1) to keep the torch.Size([1]), not null.
            loss_rgb = F.binary_cross_entropy_with_logits(pred_video_rgb.squeeze(dim=-1), label.float())
            loss_flow = F.binary_cross_entropy_with_logits(pred_video_flow.squeeze(dim=-1), single_label.float())

            loss = (loss_rgb + loss_flow) / 2

            self.save_multi_single_log([pred_video_rgb, pred_video_flow], label, single_label, loss)
        
        elif self.fusion == 'sum_loss':

            pred_sum = (pred_video_rgb.repeat_interleave(30) + pred_video_flow.squeeze(dim=-1)) / 2

            loss = F.binary_cross_entropy_with_logits(pred_sum, single_label.float())

            self.save_log([pred_sum], single_label, loss)

        return loss
    
    def save_log(self, pred_list: list, label: torch.Tensor, loss):

        if self.training:

            if self.fusion == 'different_loss':

                pred_video_rgb = pred_list[0]
                pred_video_flow = pred_list[1]

                pred_video_rgb_sigmoid = torch.sigmoid(pred_video_rgb).squeeze(dim=-1)
                pred_video_flow_sigmoid = torch.sigmoid(pred_video_flow).squeeze(dim=-1)

                # video rgb metrics
                accuracy = binary_accuracy(pred_video_rgb_sigmoid, label)
                f1_score = binary_f1_score(pred_video_rgb_sigmoid, label)
                auroc = binary_auroc(pred_video_rgb_sigmoid, label)
                cohen_kappa = binary_cohen_kappa(pred_video_rgb_sigmoid, label)
                cm = binary_confusion_matrix(pred_video_rgb_sigmoid, label)

                # log to tensorboard
                self.log_dict({'train_loss': loss,
                            'train_rgb_acc': accuracy,
                            'train_rgb_f1_score': f1_score, 
                            'train_rgb_auroc': auroc, 
                            'train_rgb_cohen_kappa': cohen_kappa, 
                            })
                
                # video optical metrics
                accuracy = binary_accuracy(pred_video_flow_sigmoid, label)
                f1_score = binary_f1_score(pred_video_flow_sigmoid, label)
                auroc = binary_auroc(pred_video_flow_sigmoid, label)
                cohen_kappa = binary_cohen_kappa(pred_video_flow_sigmoid, label)
                cm = binary_confusion_matrix(pred_video_flow_sigmoid, label)

                # log to tensorboard
                self.log_dict({'train_loss': loss,
                            'train_flow_acc': accuracy,
                            'train_flow_f1_score': f1_score, 
                            'train_flow_auroc': auroc, 
                            'train_flow_cohen_kappa': cohen_kappa, 
                            })
                
            elif self.fusion == 'sum_loss':

                preds = pred_list[0]

                # when torch.size([1]), not squeeze.
                if preds.size()[0] != 1 or len(preds.size()) != 1:
                    preds = preds.squeeze(dim=-1)
                    pred_sigmoid = torch.sigmoid(preds)
                else:
                    pred_sigmoid = torch.sigmoid(preds)

                # video rgb metrics
                accuracy = binary_accuracy(pred_sigmoid, label)
                f1_score = binary_f1_score(pred_sigmoid, label)
                auroc = binary_auroc(pred_sigmoid, label)
                cohen_kappa = binary_cohen_kappa(pred_sigmoid, label)
                cm = binary_confusion_matrix(pred_sigmoid, label)

                # log to tensorboard
                self.log_dict({'train_loss': loss,
                            'train_rgb_acc': accuracy,
                            'train_rgb_f1_score': f1_score, 
                            'train_rgb_auroc': auroc, 
                            'train_rgb_cohen_kappa': cohen_kappa, 
                            })
            
        else:
            
            if self.fusion == 'different_loss':

                pred_video_rgb = pred_list[0]
                pred_video_flow = pred_list[1]

                pred_video_rgb_sigmoid = torch.sigmoid(pred_video_rgb).squeeze(dim=-1)
                pred_video_flow_sigmoid = torch.sigmoid(pred_video_flow).squeeze(dim=-1)

                # video rgb metrics
                accuracy = binary_accuracy(pred_video_rgb_sigmoid, label)
                f1_score = binary_f1_score(pred_video_rgb_sigmoid, label)
                auroc = binary_auroc(pred_video_rgb_sigmoid, label)
                cohen_kappa = binary_cohen_kappa(pred_video_rgb_sigmoid, label)
                cm = binary_confusion_matrix(pred_video_rgb_sigmoid, label)

                # log to tensorboard
                self.log_dict({'val_loss': loss,
                            'val_rgb_acc': accuracy,
                            'val_rgb_f1_score': f1_score, 
                            'val_rgb_auroc': auroc, 
                            'val_rgb_cohen_kappa': cohen_kappa, 
                            })
                
                # video optical metrics
                accuracy = binary_accuracy(pred_video_flow_sigmoid, label)
                f1_score = binary_f1_score(pred_video_flow_sigmoid, label)
                auroc = binary_auroc(pred_video_flow_sigmoid, label)
                cohen_kappa = binary_cohen_kappa(pred_video_flow_sigmoid, label)
                cm = binary_confusion_matrix(pred_video_flow_sigmoid, label)

                # log to tensorboard
                self.log_dict({'val_loss': loss,
                            'val_flow_acc': accuracy,
                            'val_flow_f1_score': f1_score, 
                            'val_flow_auroc': auroc, 
                            'val_flow_cohen_kappa': cohen_kappa, 
                            })
                
            elif self.fusion == 'sum_loss':

                preds = pred_list[0]

                # when torch.size([1]), not squeeze.
                if preds.size()[0] != 1 or len(preds.size()) != 1:
                    preds = preds.squeeze(dim=-1)
                    pred_sigmoid = torch.sigmoid(preds)
                else:
                    pred_sigmoid = torch.sigmoid(preds)

                # video rgb metrics
                accuracy = binary_accuracy(pred_sigmoid, label)
                f1_score = binary_f1_score(pred_sigmoid, label)
                auroc = binary_auroc(pred_sigmoid, label)
                cohen_kappa = binary_cohen_kappa(pred_sigmoid, label)
                cm = binary_confusion_matrix(pred_sigmoid, label)

                # log to tensorboard
                self.log_dict({'val_loss': loss,
                            'val_rgb_acc': accuracy,
                            'val_rgb_f1_score': f1_score, 
                            'val_rgb_auroc': auroc, 
                            'val_rgb_cohen_kappa': cohen_kappa, 
                            })

    def save_multi_single_log(self, pred_list: list, label: torch.Tensor, single_label: torch.Tensor, loss):

        if self.training:

            if self.fusion == 'different_loss':

                pred_video_rgb = pred_list[0]
                pred_video_flow = pred_list[1]

                pred_video_rgb_sigmoid = torch.sigmoid(pred_video_rgb).squeeze(dim=-1)
                pred_video_flow_sigmoid = torch.sigmoid(pred_video_flow).squeeze(dim=-1)

                # video rgb metrics
                accuracy = binary_accuracy(pred_video_rgb_sigmoid, label)
                f1_score = binary_f1_score(pred_video_rgb_sigmoid, label)
                auroc = binary_auroc(pred_video_rgb_sigmoid, label)
                cohen_kappa = binary_cohen_kappa(pred_video_rgb_sigmoid, label)
                cm = binary_confusion_matrix(pred_video_rgb_sigmoid, label)

                # log to tensorboard
                self.log_dict({'train_loss': loss,
                            'train_rgb_acc': accuracy,
                            'train_rgb_f1_score': f1_score, 
                            'train_rgb_auroc': auroc, 
                            'train_rgb_cohen_kappa': cohen_kappa, 
                            })
                
                # single optical metrics
                accuracy = binary_accuracy(pred_video_flow_sigmoid, single_label)
                f1_score = binary_f1_score(pred_video_flow_sigmoid, single_label)
                auroc = binary_auroc(pred_video_flow_sigmoid, single_label)
                cohen_kappa = binary_cohen_kappa(pred_video_flow_sigmoid, single_label)
                cm = binary_confusion_matrix(pred_video_flow_sigmoid, single_label)

                # log to tensorboard
                self.log_dict({'train_loss': loss,
                            'train_flow_acc': accuracy,
                            'train_flow_f1_score': f1_score, 
                            'train_flow_auroc': auroc, 
                            'train_flow_cohen_kappa': cohen_kappa, 
                            })
            
        else:
            
            if self.fusion == 'different_loss':

                pred_video_rgb = pred_list[0]
                pred_video_flow = pred_list[1]

                pred_video_rgb_sigmoid = torch.sigmoid(pred_video_rgb).squeeze(dim=-1)
                pred_video_flow_sigmoid = torch.sigmoid(pred_video_flow).squeeze(dim=-1)

                # video rgb metrics
                accuracy = binary_accuracy(pred_video_rgb_sigmoid, label)
                f1_score = binary_f1_score(pred_video_rgb_sigmoid, label)
                auroc = binary_auroc(pred_video_rgb_sigmoid, label)
                cohen_kappa = binary_cohen_kappa(pred_video_rgb_sigmoid, label)
                cm = binary_confusion_matrix(pred_video_rgb_sigmoid, label)

                # log to tensorboard
                self.log_dict({'val_loss': loss,
                            'val_rgb_acc': accuracy,
                            'val_rgb_f1_score': f1_score, 
                            'val_rgb_auroc': auroc, 
                            'val_rgb_cohen_kappa': cohen_kappa, 
                            })
                
                # single optical metrics
                accuracy = binary_accuracy(pred_video_flow_sigmoid, single_label)
                f1_score = binary_f1_score(pred_video_flow_sigmoid, single_label)
                auroc = binary_auroc(pred_video_flow_sigmoid, single_label)
                cohen_kappa = binary_cohen_kappa(pred_video_flow_sigmoid, single_label)
                cm = binary_confusion_matrix(pred_video_flow_sigmoid, single_label)

                # log to tensorboard
                self.log_dict({'val_loss': loss,
                            'val_flow_acc': accuracy,
                            'val_flow_f1_score': f1_score, 
                            'val_flow_auroc': auroc, 
                            'val_flow_cohen_kappa': cohen_kappa, 
                            })
                