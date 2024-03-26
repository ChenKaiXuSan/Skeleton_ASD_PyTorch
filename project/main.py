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
2023-10-29	KX.C	add the lr monitor, and fast dev run to trainer.

"""

import os, logging
from typing import Any
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

import hydra
from cross_validation import DefineCrossValidation

def train(hparams, dataset_idx, fold):
    seed_everything(42, workers=True)

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
        monitor="val/loss",
        patience=5,
        mode="min",
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

    for fold, dataset_value in fold_dataset_idx.items():
        logging.info("#" * 50)
        logging.info("Start train fold: {}".format(fold))
        logging.info("#" * 50)

        train(config, dataset_value, fold)

        logging.info("#" * 50)
        logging.info("finish train fold: {}".format(fold))
        logging.info("#" * 50)

    logging.info("#" * 50)
    logging.info("finish train all fold")
    logging.info("#" * 50)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
