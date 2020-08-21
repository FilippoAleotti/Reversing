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

from dataloader.utils import image_loader
from dataloader import augment


class Loader(data.Dataset):
    """Dataloader for Middlebury dataset.
    This dataset is used only for testing purposes.
    """

    def __init__(
        self, left, right, params, mode, augmentation_params=None,
    ):
        """Create the Middlebury loader.
        Args:
            left: path to left images
            right: path to right images
            params: dictionary of parameters
            mode: training or testing mode
            augmentation_params: dictionary with augmentation configuration
        """
        self.left = left
        self.right = right
        self.loader = image_loader
        self.params = params
        self.mode = mode
        if mode != "test":
            raise ValueError("Expected test mode for Middlebury/ETH3D")
        self.augmentation_params = augmentation_params
        self.transformations = self._set_transformations()

    def __getitem__(self, index):
        """Get next item.
        Args:
            index: current image index
        Return:
            left: numpy array with shape HxWx3 array, with the left image
            right: numpy array with shape HxWx3 array, with the right image
            (h, w): list with original height and width of the image
            name: name of the image
        """
        left = os.path.join(self.params["datapath"], self.left[index])
        right = os.path.join(self.params["datapath"], self.right[index])

        left = self.loader(left)
        right = self.loader(right)
        h, w, _ = left.shape
        name = os.path.basename(self.left[index].replace("/im0.png", "")) + ".png"

        top_pad = self.params["final_h"] - h
        left_pad = self.params["final_w"] - w
        left = np.lib.pad(left, ((top_pad, 0), (0, left_pad), (0, 0)), mode="edge")
        right = np.lib.pad(right, ((top_pad, 0), (0, left_pad), (0, 0)), mode="edge")

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
