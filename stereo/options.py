from __future__ import absolute_import, division, print_function

import os
import argparse
import torch
from utils import general

file_dir = os.path.dirname(__file__)


class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Reversing the Cycle options")

        # general
        parser.add_argument(
            "--gpu_ids", type=str, help="id(s) of the gpu to use", default="0"
        )
        parser.add_argument(
            "--mode", type=str, default="train", choices=["train", "test"]
        )
        parser.add_argument("--ckpt", default=None, help="path to a checkpoint")
        parser.add_argument("--maxdisp", type=int, default=192, help="maxium disparity")
        parser.add_argument("--model", help="select model", required=True)
        parser.add_argument(
            "--datapath", type=str, default=None, help="path to rgb images"
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        parser.add_argument(
            "--dataset",
            type=str,
            help="dataset to use",
            default="KITTI",
            choices=["KITTI", "MIDDLEBURY", "ETH3D", "DS",],
        )

        # train
        parser.add_argument(
            "--proxy", help="path to proxy", type=str, default=None,
        )
        parser.add_argument(
            "--epochs", type=int, help="number of epochs to train", default=11
        )
        parser.add_argument("--savemodel", default="./logs", help="save model")
        parser.add_argument("--batch", type=int, default=2, help="batch size")
        parser.add_argument("--crop_h", type=int, default=192, help="crop height")
        parser.add_argument("--crop_w", type=int, default=640, help="crop width")
        parser.add_argument(
            "--print_freq",
            type=int,
            help="step between logs in tensorboard",
            default=200,
        )
        parser.add_argument(
            "--training_port",
            default="6006",
            type=str,
            help="port used by TensorBoard to display training outcomes",
        )
        parser.add_argument(
            "--milestone",
            type=int,
            help="epochs at which learning rate is divided by decay factor",
        )
        parser.add_argument(
            "--decay_factor", type=float, help="decaying factor", default=0.5
        )
        parser.add_argument(
            "--loss_weights",
            type=str,
            help="weight for each scale (comma separated, from lowest scale up to full scale)",
        )
        parser.add_argument(
            "--initial_learning_rate",
            type=float,
            default=1e-4,
            help="initial learning rate",
        )
        parser.add_argument(
            "--filename",
            type=str,
            help="filename to use (if required by the dataset)",
            default="filenames/kitti_train_files.txt",
        )
        parser.add_argument(
            "--rgb_ext",
            type=str,
            help="extension of rgb in your dataset (if required)",
            default=".jpg",
        )

        # test
        parser.add_argument(
            "--final_h", type=int, default=384, help="height after pad in test",
        )
        parser.add_argument(
            "--final_w", type=int, default=1280, help="width after pad in test",
        )
        parser.add_argument(
            "--results", type=str, default="./results", help="test results folder"
        )
        parser.add_argument(
            "--qualitative",
            action="store_true",
            help="save prediction as qualitative using a color map",
        )
        parser.add_argument(
            "--cmap",
            type=str,
            default="magma",
            help="colormap to use",
            choices=["magma", "gray", "jet", "kitti"],
        )

        self.parser = parser
        self.train_params = None
        self.loss_params = None
        self.augmentation_params = None
        self.log_params = None
        self.dataset_params = None
        self.options = None
        self.general_params = None
        self.test_params = None
        self.padding_params = None

    def _parse(self):
        """Parse arguments
        """
        if self.options is None:
            self.options = self.parser.parse_args()

    def parse_general_params(self):
        """Obtain general params
        """
        self._parse()
        gpus = general.parse_gpu_ids(self.options.gpu_ids)
        cuda = len(gpus) > 0 and torch.cuda.is_available()
        self.general_params = {
            "maxdisp": self.options.maxdisp,
            "cuda": cuda,
            "seed": self.options.seed,
            "gpu_ids": gpus,
            "model": self.options.model,
            "mode": self.options.mode,
        }

    def parse_train_params(self):
        """Obtain training params
        """
        self._parse()
        self.dataset_params = {
            "datapath": self.options.datapath,
            "proxy": self.options.proxy,
            "dataset": self.options.dataset,
            "filename": self.options.filename,
            "rgb_ext": self.options.rgb_ext,
        }

        self.augmentation_params = {
            "crop_height": self.options.crop_h,
            "crop_width": self.options.crop_w,
            "probability_swap": 0.5,
        }

        self.loss_params = {"weights": general.parse_weights(self.options.loss_weights)}

        self.log_params = {
            "max_el": 1,
            "print_freq": self.options.print_freq,
            "training_port": self.options.training_port,
        }

        self.train_params = {
            "epochs": self.options.epochs,
            "savemodel": self.options.savemodel,
            "model": self.options.model,
            "batch": self.options.batch,
            "milestone": self.options.milestone,
            "decay_factor": self.options.decay_factor,
            "ckpt": self.options.ckpt,
            "initial_learning_rate": self.options.initial_learning_rate,
        }

    def parse_test_params(self):
        """Obtain testing params
        """
        self._parse()
        self.test_params = {
            "results": self.options.results,
            "model": self.options.model,
            "ckpt": self.options.ckpt,
            "dataset": self.options.dataset,
            "datapath": self.options.datapath,
            "qualitative": self.options.qualitative,
            "cmap": self.options.cmap,
            "rgb_ext": self.options.rgb_ext,
        }

        self.padding_params = {
            "final_h": self.options.final_h,
            "final_w": self.options.final_w,
        }
