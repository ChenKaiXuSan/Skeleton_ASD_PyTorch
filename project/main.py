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

import os, logging, time, sys, json
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

from sklearn.model_selection import StratifiedGroupKFold, train_test_split, GroupKFold
from pathlib import Path


def train(hparams, dataset_idx, fold):

    seed_everything(42, workers=True)

    classification_module = GaitCycleLightningModule(hparams)

    data_module = WalkDataModule(hparams, dataset_idx)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.train.log_path),
        name=fold
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
        patience=10,
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

    trainer.validate(classification_module, data_module, ckpt_path="best")

def define_cross_validation(video_path: str, K: int = 5):
    """define cross validation first, with the K.
    #! the 1 fold and K fold should return the same format.
    train = {
        0: {
            "disease1": [patient1_idx, patient2_idx, ...],
            "disease2": [patient1_idx, patient2_idx, ...],
    }
    val = {
        0: {
            "disease1": [patient1_idx, patient2_idx, ...],
            "disease2": [patient1_idx, patient2_idx, ...],
    }

    Args:
        video_path (str): the index of the video path, in .json format.
        K (int, optional): crossed number of validation. Defaults to 5, can be 1 or K.

    Returns:
        list: the format like upper.
    """

    _path = Path(video_path)

    # define the cross validation
    if K > 1:
        ans_Dict = {}

        # process one disease in one loop.
        for disease in _path.iterdir():
            disease_List = []

            X = []  # patient index
            groups = []  # different patient groups

            if disease.name != "log":
                patient_list = sorted(list(disease.iterdir()))
                name_map = set()

                for p in patient_list:
                    name, _ = p.name.split("-")
                    name_map.add(name)

                element_to_num = {element: idx for idx, element in enumerate(name_map)}

                for i in range(len(patient_list)):
                    name, _ = patient_list[i].name.split("-")

                    X.append(i)

                    groups.append(element_to_num[name])

                sgkf = GroupKFold(n_splits=K)

                for i, (train_index, test_index) in enumerate(
                    sgkf.split(X=X, groups=groups)
                ):
                    train_idx = [patient_list[i] for i in train_index]
                    val_idx = [patient_list[i] for i in test_index]

                    # store one fold in one disease
                    disease_List.append([train_idx, val_idx])

                ans_Dict[disease.name] = disease_List
                # TODO: 需要把cv的结果可视化一下，看看有没有混用

        final_ans_dict = {}

        # * convert the disease:fold:train/val
        # * fold: train/val: disease: index

        for i in range(K):

            train_dict = {}
            val_dict = {}

            for d, f in ans_Dict.items():
                for j in range(len(f)):
                    train_dict[d] = f[j][0]
                    val_dict[d] = f[j][1]

            final_ans_dict[i] = [train_dict, val_dict]

        # writ to csv file 
        with open('./misc/data_distribution.json', 'w') as f:
            json.dump(final_ans_dict, f)

        return final_ans_dict

    # only have 1 fold
    else:
        train_idx = {}
        val_idx = {}
        for disease in _path.iterdir():
            if disease.name != "log":
                sample_list = list(disease.iterdir())
                train, val = train_test_split(
                    range(len(sample_list)), test_size=0.2, random_state=42
                )

                # here load the video path by index.
                train_idx[disease.name] = {0: [sample_list[i] for i in train]}
                val_idx[disease.name] = {0: [sample_list[i] for i in val]}

        return {0: [train_idx, val_idx]}


@hydra.main(
    version_base=None,
    config_path="/workspace/skeleton/configs",
    config_name="config.yaml",
)
def init_params(config):

    # DATE = str(time.localtime().tm_mon) + str(time.localtime().tm_mday)

    _gait_seg_path = config.data.gait_seg_data_path

    # set the version
    # uniform_temporal_subsample_num = config.train.uniform_temporal_subsample_num
    # clip_duration = config.train.clip_duration
    # config.train.version = "_".join(
    #     [DATE, str(clip_duration), str(uniform_temporal_subsample_num)]
    # )

    # * we need prepare the cross validation dataset index first.
    fold_dataset_idx = define_cross_validation(_gait_seg_path, config.train.fold)

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
