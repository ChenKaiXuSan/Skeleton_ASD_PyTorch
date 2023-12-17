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

import os, logging, time, sys, json, yaml, csv, shutil, copy
from typing import Any
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

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


class DefineCrossValidation:
    """process:
    cross validation > over/under sampler > train/val split
    fold: [train/val]: [path]
    """

    def __init__(self, config) -> None:
        self.video_path = Path(config.data.gait_seg_data_path)
        self.gait_seg_idx_path = Path(config.data.gait_seg_index_data_path)
        self.K = config.train.fold
        self.sampler = config.data.sampling

    @staticmethod
    def random_sampler(X: list, y: list, train_idx: list, val_idx: list, sampler):
        # train
        train_mapped_path = []
        new_X_path = [X[i] for i in train_idx]

        sampled_X, sampled_y = sampler.fit_resample(
            [[i] for i in range(len(new_X_path))], [y[i] for i in train_idx]
        )

        # map sampled_X to new_X_path
        for i in sampled_X:
            train_mapped_path.append(new_X_path[i[0]])

        # val
        val_mapped_path = []
        new_X_path = [X[i] for i in val_idx]

        sampled_X, sampled_y = sampler.fit_resample(
            [[i] for i in range(len(new_X_path))], [y[i] for i in val_idx]
        )

        # map
        for i in sampled_X:
            val_mapped_path.append(new_X_path[i[0]])

        return train_mapped_path, val_mapped_path

    @staticmethod
    def process_cross_validation(video_path):
        _path = video_path

        X = []  # patient index
        y = []  # patient class index
        groups = []  # different patient groups
        # process one disease in one loop.
        for disease in _path.iterdir():
            if disease.name != "log":
                patient_list = sorted(list(disease.iterdir()))
                name_map = set()

                for p in patient_list:
                    name, _ = p.name.split("-")
                    name_map.add(name)

                element_to_num = {element: idx for idx, element in enumerate(name_map)}

                for i in range(len(patient_list)):
                    name, _ = patient_list[i].name.split("-")
                    # load the video tensor from json file
                    with open(patient_list[i]) as f:
                        file_info_dict = json.load(f)

                    label = file_info_dict["label"]

                    X.append(patient_list[i])  # true path in Path
                    y.append(label)  # label, 0, 1, 2
                    groups.append(element_to_num[name])  # number of different patient

        return X, y, groups

    def make_val_dataset(self, val_dataset_idx: list, fold: int):
        temp_val_path = self.gait_seg_idx_path / self.sampler / str(fold)
        val_idx = val_dataset_idx

        shutil.rmtree(temp_val_path, ignore_errors=True)

        for path in val_idx:
            with open(path) as f:
                file_info_dict = json.load(f)

            video_name = file_info_dict["video_name"]
            video_path = file_info_dict["video_path"]
            video_disease = file_info_dict["disease"]

            if not (temp_val_path / video_disease).exists():
                (temp_val_path / video_disease).mkdir(parents=True, exist_ok=False)

            shutil.copy(
                video_path, temp_val_path / video_disease / (video_name + ".mp4")
            )

        return temp_val_path

    def prepare(self):
        """define cross validation first, with the K.
        #! the 1 fold and K fold should return the same format.
        fold: [train/val]: [path]

        Args:
            video_path (str): the index of the video path, in .json format.
            K (int, optional): crossed number of validation. Defaults to 5, can be 1 or K.

        Returns:
            list: the format like upper.
        """
        K = self.K

        ans_fold = {}

        # define the cross validation
        X, y, groups = self.process_cross_validation(self.video_path)

        sgkf = StratifiedGroupKFold(n_splits=K)

        for i, (train_index, test_index) in enumerate(
            sgkf.split(X=X, y=y, groups=groups)
        ):
            if self.sampler in ["over", "under"]:
                if self.sampler == "over":
                    ros = RandomOverSampler(random_state=42)
                elif self.sampler == "under":
                    ros = RandomUnderSampler(random_state=42)

                train_mapped_path, val_mapped_path = self.random_sampler(
                    X, y, train_index, test_index, ros
                )

            else:
                train_mapped_path = [X[i] for i in train_index]
                val_mapped_path = [X[i] for i in test_index]

            # make the val data path
            val_mapped_path = self.make_val_dataset(val_mapped_path, i)

            ans_fold[i] = [train_mapped_path, val_mapped_path]

        return ans_fold, X, y, groups

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        if not os.path.exists(self.gait_seg_idx_path/ self.sampler):
            fold_dataset_idx, *_ = self.prepare()

            json_fold_dataset_idx = copy.deepcopy(fold_dataset_idx)

            for k, v in fold_dataset_idx.items():
                
                # train
                train_dataset_idx = v[0]

                json_fold_dataset_idx[k][0] = [str(i) for i in train_dataset_idx]

                # val
                val_dataset_idx = v[1]
                json_fold_dataset_idx[k][1] = str(val_dataset_idx)

            with open((self.gait_seg_idx_path / self.sampler / "index.json"), "w") as f:
                json.dump(json_fold_dataset_idx, f)


        elif os.path.exists(self.gait_seg_idx_path):
            with open(self.gait_seg_idx_path / self.sampler / "index.json", "r") as f:
                fold_dataset_idx = json.load(f)

            for k, v in fold_dataset_idx.items():
                # train
                train_dataset_idx = v[0]
                fold_dataset_idx[k][0] = [Path(i) for i in train_dataset_idx]

                # val
                val_dataset_idx = v[1]
                fold_dataset_idx[k][1] = Path(val_dataset_idx)
                
        else:
            raise ValueError(
                "the gait seg idx path is not exist, please check the path."
            )

        return fold_dataset_idx


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
