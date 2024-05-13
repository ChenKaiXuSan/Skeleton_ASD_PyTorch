"""
File: main.py
Project: project
Created Date: 2023-10-19 02:29:35
Author: chenkaixu
-----
Comment:
 
Have a good code time!
-----
Last Modified: Thursday October 19th 2023 2:29:35 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

04-04-2024	Kaixu Chen	add save inference method. now it can save the pred/label to the disk, for the further analysis.
2023-10-29	KX.C	add the lr monitor, and fast dev run to trainer.

"""

import os, logging
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

# callbacks
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    RichModelSummary,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from dataloader.data_loader import WalkDataModule
from train import GaitCycleLightningModule
from train_late_fusion import LateFusionModule

import hydra
from cross_validation import DefineCrossValidation

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
)

def save_inference(config, model, dataloader, fold):

    total_pred_list = []
    total_label_list = []

    dataloader.setup()
    test_dataloader = dataloader.test_dataloader()

    for i, batch in enumerate(test_dataloader):

        pred_list = []
        label_list = []

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

    # save confusion matrix 
    save_CM(pred, label, fold)

def save_CM(all_pred, all_label, fold, config):

    save_path = Path(config.train.log_path) / "CM"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    # define metrics 
    num_class = torch.unique(all_label).size(0)
    _accuracy = MulticlassAccuracy(num_class)
    _precision = MulticlassPrecision(num_class)
    _recall = MulticlassRecall(num_class)
    _f1_score = MulticlassF1Score(num_class)
    _auroc = MulticlassAUROC(num_class)
    _confusion_matrix = MulticlassConfusionMatrix(num_class, normalize="true")

    print('*' * 100)
    print('accuracy: %s' % _accuracy(all_pred, all_label))
    print('precision: %s' % _precision(all_pred, all_label))
    print('_binary_recall: %s' % _recall(all_pred, all_label))
    print('_binary_f1: %s' % _f1_score(all_pred, all_label))
    print('_aurroc: %s' % _auroc(all_pred, all_label))
    print('_confusion_matrix: %s' % _confusion_matrix(all_pred, all_label))
    print('#' * 100)

   
    # 设置字体和标题样式
    plt.rcParams.update({'font.size': 30, 'font.family': 'sans-serif'})

    # 假设的混淆矩阵数据
    confusion_matrix_data = _confusion_matrix(all_pred, all_label).cpu().numpy() * 100

    axis_labels = ['ASD', 'DHS', 'LCS_HipOA']

    # 使用matplotlib和seaborn绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_data, annot=True, fmt='.2f', cmap='Reds', xticklabels=axis_labels, yticklabels=axis_labels, vmin=0, vmax=100)
    plt.title(f'Fold {fold} (%)', fontsize=30)
    plt.ylabel('Actual Label', fontsize=30)
    plt.xlabel('Predicted Label', fontsize=30)

    plt.savefig(save_path / f'fold{fold}_confusion_matrix.png', dpi=300, bbox_inches='tight')

    print(f'save the confusion matrix into {save_path} / fold{fold}_confusion_matrix.png')

def train(hparams, dataset_idx, fold):
    """the train process for the one fold.

    Args:
        hparams (hydra): the hyperparameters.
        dataset_idx (int): the dataset index for the one fold.
        fold (int): the fold index.

    Returns:
        list: best trained model, data loader
    """    
    
    seed_everything(42, workers=True)

    if hparams.train.experiment == "late_fusion":
        classification_module = LateFusionModule(hparams)
    else:
        classification_module = GaitCycleLightningModule(hparams)

    data_module = WalkDataModule(hparams, dataset_idx)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.train.log_path),
        name=str(fold),  # here should be str type.
    )

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=100)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename="{epoch}-{val/loss:.2f}-{val/video_acc:.4f}",
        auto_insert_metric_name=False,
        monitor="val/video_acc",
        mode="max",
        save_last=False,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor="val/video_acc",
        patience=2,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        devices=[
            int(hparams.train.gpu_num),
        ],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=tb_logger,  # wandb_logger,
        check_val_every_n_epoch=1,
        callbacks=[
            progress_bar,
            rich_model_summary,
            model_check_point,
            early_stopping,
            lr_monitor,
        ],
        fast_dev_run=hparams.train.fast_dev_run,  # if use fast dev run for debug.
    )

    trainer.fit(classification_module, data_module)

    # the validate method will wirte in the same log twice, so use the test method.
    trainer.test(classification_module, data_module, ckpt_path="best")

    return classification_module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path), data_module


@hydra.main(
    version_base=None,
    config_path="/workspace/skeleton/configs",
    config_name="config.yaml",
)
def init_params(config):
    #############
    # prepare dataset index
    #############

    fold_dataset_idx = DefineCrossValidation(config)()

    logging.info("#" * 50)
    logging.info("Start train all fold")
    logging.info("#" * 50)

    #############
    # K fold
    #############
    # * for one fold, we first train/val model, then save the best ckpt preds/label into .pt file.
    
    for fold, dataset_value in fold_dataset_idx.items():
        logging.info("#" * 50)
        logging.info("Start train fold: {}".format(fold))
        logging.info("#" * 50)

        classification_module, data_module = train(config, dataset_value, fold)

        logging.info("#" * 50)
        logging.info("finish train fold: {}".format(fold))
        logging.info("#" * 50)

        save_inference(config, classification_module, data_module, fold)

    logging.info("#" * 50)
    logging.info("finish train all fold")
    logging.info("#" * 50)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
