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

from __future__ import division
import os
import torch
import torch.utils.data as data
import numpy as np
from torchvision import transforms
import cv2
from dataloader import augment
from dataloader.utils import image_loader


def gt_loader(path):
    """Load gt depth
    Args:
        path: path to 16 bit png depth
    Return:
        The float32 depth map
    """
    gt = cv2.imread(path, -1)
    gt = gt / 256.0
    gt = np.expand_dims(gt, -1)
    return gt


def proxy_loader(path):
    """Load proxy depth
    Args:
        path: path to 16 bit png depth
    Return:
        The float32 proxy depth map
    """
    try:
        proxy = cv2.imread(path, -1)
        proxy = np.expand_dims(proxy, -1)
        proxy = proxy / 256.0
    except:
        print("Cannot open proxy: " + path)
    return proxy


class Loader(data.Dataset):
    """Dataloader for KITTI dataset
    """

    def __init__(
        self,
        left,
        right,
        params,
        mode,
        proxy_left=None,
        proxy_right=None,
        augmentation_params=None,
    ):
        """Create the KITTI loader.
        Args:
            left: path to left images
            right: path to right images
            params: dictionary of parameters
            mode: training or test flag
            proxy_left: path to proxy left. Default is None
            proxy_right: path to proxy right. Default is None
            augmentation_params: dictionary with augmentation configuration
        """
        self.left = left
        self.right = right
        self.disp_L = proxy_left
        self.disp_R = proxy_right
        self.mode = mode
        self.loader = image_loader
        self.proxy_loader = proxy_loader
        self.training = mode == "training"

        if self.training == False:
            self.proxy_loader = gt_loader

        self.params = params
        self.augmentations = self._set_augmentations(augmentation_params)
        self.transformations = self._set_transformations()

    def __getitem__(self, index):
        """Get next item. The behaviour depends on the current mode (training or test)
        Args:
            index: current image index

        Return:
            In case of training this method retuns: left, right, proxy_left
            In case of test, this method returns: left, right, (h, w), name
            Where:
                left: numpy array with shape HxWx3 array, with the left image
                right: numpy array with shape HxWx3 array, with the right image
                gt: numpy array with shape HxWx1 containing the ground-truth depth
                proxy: numpy array with shape HxWx1 containing the proxy depth value
                (h, w): list with original height and width of the image
                name: name of the image
        """
        if self.training:
            left = os.path.join(self.params["datapath"], self.left[index])
            right = os.path.join(self.params["datapath"], self.right[index])

            left = self.loader(left)
            right = self.loader(right)
            proxy_left = os.path.join(self.params["proxy"], self.disp_L[index])
            proxy_right = os.path.join(self.params["proxy"], self.disp_R[index])

            proxy_left = self.proxy_loader(proxy_left)
            proxy_right = self.proxy_loader(proxy_right)

            # apply augmentations
            if self.augmentations is not None:
                [left, right], [proxy_left, proxy_right] = self.augmentations(
                    [left, right], [proxy_left, proxy_right]
                )

            # apply transformations
            left = self.transformations["source"](left)
            right = self.transformations["source"](right)

            proxy_left = self.transformations["target"](proxy_left)

            return left, right, proxy_left
        else:
            left = self.left[index]
            right = self.right[index]
            left = self.loader(left)
            right = self.loader(right)
            h, w, _ = left.shape
            name = str(index).zfill(6) + "_10.png"

            if self.augmentations is not None:
                left, right = self.augmentations([left, right])
            top_pad = self.params["final_h"] - h
            left_pad = self.params["final_w"] - w
            left = np.lib.pad(left, ((top_pad, 0), (0, left_pad), (0, 0)), mode="edge")
            right = np.lib.pad(
                right, ((top_pad, 0), (0, left_pad), (0, 0)), mode="edge"
            )

            left = self.transformations["source"](left)
            right = self.transformations["source"](right)
            return left, right, (h, w), name

    def __len__(self):
        """Get the number of elements
        Return:
            number of elements
        """
        return len(self.left)

    def _set_transformations(self):
        """Transform inputs and eventually target in normalised tensors
        Return:
            A dictionary with transformation to apply
        """
        input_transformation = augment.ComposeTransformation(
            [
                augment.ArrayToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            ]
        )
        target_transformation = augment.ComposeTransformation(
            [augment.ArrayToTensor(),]
        )

        trasformations = {
            "source": input_transformation,
            "target": target_transformation,
        }
        return trasformations

    def _set_augmentations(self, augmentation_params):
        """Define augmentation to apply.
        Args:
            augmentation_params: dictionary with augmentation configuration

        Return:
            Composition of augmentations
        """
        if augmentation_params is None:
            return None
        co_transform = None

        if self.training:
            crop_w = augmentation_params["crop_width"]
            crop_h = augmentation_params["crop_height"]
            prob_swap = augmentation_params["probability_swap"]

            scaling = augment.RandomCrop((crop_h, crop_w))
            flip_method = augment.FlipLeftRight(prob_swap)
            co_transform = augment.Compose(
                [flip_method, scaling, augment.ColorAugmentation()]
            )

        return co_transform
