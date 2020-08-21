# Copyright 2020 Filippo Aleotti, Fabio Tosi, Li Zhang, Matteo Poggi, Stefano Mattoccia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import general, tensor
from models import network_factory
from dataloader import augment
from torchvision import transforms

parser = argparse.ArgumentParser(description="Reversing the cycle: single shot")

# general
parser.add_argument("--gpu_ids", type=str, help="id(s) of the gpu to use", default="0")
parser.add_argument("--ckpt", help="path to checkpoint", required=True)
parser.add_argument("--maxdisp", type=int, default=192, help="maxium disparity")
parser.add_argument("--model", help="stereo network to use", required=True)
parser.add_argument(
    "--left", type=str, help="path to left image(s) [space separated]", required=True
)
parser.add_argument(
    "--right", type=str, help="path to right image(s) [space separated]", required=True
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
# test
parser.add_argument("--final_h", type=int, default=384, help="height after pad in test")
parser.add_argument("--final_w", type=int, default=1280, help="width after pad in test")
parser.add_argument("--results", type=str, default="./artifacts", help="result folder")
parser.add_argument(
    "--qualitative", action="store_true", help="save colored maps instead of 16bit"
)
parser.add_argument(
    "--cmap",
    type=str,
    default="magma",
    help="colormap to use",
    choices=["magma", "gray", "jet", "kitti"],
)
parser.add_argument(
    "--maxval", type=int, default=-1, help="max value in kitti colormap"
)

args = parser.parse_args()

gpus = general.parse_gpu_ids(args.gpu_ids)
args.cuda = len(gpus) > 0 and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def _parse(names):
    """Split a string with space separated valus.
    Args:

    Return:
        a list where each element is a non-empty value of the original list
    """
    imgs = names.split(" ")
    imgs = [x.strip() for x in imgs if x.strip()]
    return imgs


def run_single_shot(network):
    """ Generate depth for a single (or a list of) example.
    Args:
        network: pre-trained stereo model
    """

    test_params = {
        "results": args.results,
        "model": args.model,
        "lefts": _parse(args.left),
        "rights": _parse(args.right),
        "qualitative": args.qualitative,
        "maxval": args.maxval,
        "cmap": args.cmap,
    }
    padding_params = {
        "final_h": args.final_h,
        "final_w": args.final_w,
    }

    network.eval()
    transformation = augment.ComposeTransformation(
        [
            augment.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ]
    )
    with tqdm(total=len(test_params["lefts"])) as pbar:
        for (left_i, right_i) in zip(test_params["lefts"], test_params["rights"]):
            if not os.path.exists(left_i):
                print("missing left image:{}".format(left_i))
                continue
            if not os.path.exists(right_i):
                print("missing right image:{}".format(right_i))
                continue

            left_img = cv2.imread(left_i)
            right_img = cv2.imread(right_i)

            if left_img.shape != right_img.shape:
                raise ValueError("Left and right images have different shapes")

            h, w, _ = left_img.shape
            top_pad = padding_params["final_h"] - h
            left_pad = padding_params["final_w"] - w

            # add padding
            left_img = np.lib.pad(
                left_img, ((top_pad, 0), (0, left_pad), (0, 0)), mode="edge"
            )
            right_img = np.lib.pad(
                right_img, ((top_pad, 0), (0, left_pad), (0, 0)), mode="edge"
            )

            # transorm to tensor
            left = transformation(left_img)
            right = transformation(right_img)

            # create batch
            left = torch.unsqueeze(left, 0)
            right = torch.unsqueeze(right, 0)

            name = "disp_" + os.path.basename(left_i)

            if args.cuda:
                # loading images on GPU
                left = torch.FloatTensor(left).cuda()
                right = torch.FloatTensor(right).cuda()
                left, right = Variable(left), Variable(right)

            # make prediction
            with torch.no_grad():
                output = network(left, right)
                output = torch.squeeze(output)
                output = torch.nn.functional.relu(output)
                output = output.data.cpu().numpy()
                extension = "." + name.split(".")[-1]
                name = name.replace(extension, "")

            # remove padding
            if left_pad == 0:
                final_output = output[top_pad:, :]
            else:
                final_output = output[top_pad:, :-left_pad]

            if final_output.shape[0] != h or final_output.shape[1] != w:
                raise ValueError("Problems in cropping final predictions")

            destination = os.path.join(
                test_params["results"], test_params["model"], "{}", name + ".png"
            )

            # saving predictions
            if test_params["qualitative"]:
                min_value = final_output.min()
                max_value = final_output.max()
                final_output = (final_output - min_value) / (max_value - min_value)
                final_output *= 255.0
                general.save_color(
                    destination.format("qualitative"),
                    final_output,
                    cmap=test_params["cmap"],
                    params={"maxval": test_params["maxval"]},
                )
            else:
                general.save_kitti_disp(destination.format("16bit"), final_output)

            pbar.update(1)
    print("Done! Predictions saved in {} folder".format(test_params["results"]))


if __name__ == "__main__":
    print("=> model: {}".format(args.model))
    print("=> checkpoint: {}".format(args.ckpt))
    if not os.path.exists(args.ckpt):
        raise ValueError("Checkpoint not found!")

    model = network_factory.get_network(args.model)(
        {"maxdisp": args.maxdisp, "imagenet_pt": False}
    )

    if args.cuda:
        print("=> selected gpu(s) with ids {}".format(*gpus))
        model = nn.DataParallel(model)
        model.cuda()

    print(
        "=> Number of model parameters: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict["state_dict"], strict=True)
    print("EPOCHS: {}".format(state_dict["epoch"]))
    run_single_shot(model)
