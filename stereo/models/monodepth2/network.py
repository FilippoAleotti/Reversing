# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

# Code from https://github.com/nianticlabs/monodepth2

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict
from models.monodepth2.layers import *


class MonoDepth2(nn.Module):
    """Monodepth2 network
    """

    def __init__(self, params):
        """Create the network.
        Args:
            params: dictionary with newtork params
        """
        super(MonoDepth2, self).__init__()
        self.maxdisp = params["maxdisp"]
        self.encoder = ResnetEncoder(num_layers=18, num_input_images=2,)
        num_ch_enc = self.encoder.num_ch_enc
        self.decoder = DepthDecoder(num_ch_enc, scales=range(3), maxdisp=self.maxdisp)

    def forward(self, left, right):
        """Run the network.
        Args:
            left: left image of the stereo pair. Tensor with shape HxWx3 or BxHxWx3
            right: right image of the stereo pair. Tensor with shape HxWx3 or BxHxWx3
        Return:
            A tensor with shape BxHxWx1 containing disparity values
        """
        if len(left.shape) == 3 and len(right.shape) == 3:
            x = torch.stack([left, right], dim=1)
        elif len(left.shape) == 4 and len(right.shape) == 4:
            x = torch.cat([left, right], dim=1)
        else:
            raise ValueError("not expected shape for inputs")
        _, _, h, w = x.shape
        features = self.encoder(x)
        return self.decoder(features, (h, w))


class DepthDecoder(nn.Module):
    def __init__(
        self,
        num_ch_enc,
        maxdisp,
        scales=range(4),
        num_output_channels=1,
        use_skips=True,
    ):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "nearest"
        self.scales = scales
        self.maxdisp = maxdisp
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(
                self.num_ch_dec[s], self.num_output_channels
            )

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.activation = nn.Sigmoid()

    def forward(self, input_features, size):
        self.outputs = {}
        h, w = size

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = (
                    self.activation(self.convs[("dispconv", i)](x)) * self.maxdisp
                )

        disp0 = self.outputs[("disp", 0)]
        if self.training:
            disp2 = F.interpolate(
                self.outputs[("disp", 2)], [h, w], mode="bilinear", align_corners=True
            )
            disp1 = F.interpolate(
                self.outputs[("disp", 1)], [h, w], mode="bilinear", align_corners=True
            )
            return disp2, disp1, disp0
        else:
            return disp0


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=2):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, num_input_images=2):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[
        num_layers
    ]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        if num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet layers".format(num_layers)
            )

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, num_input_images)
        else:
            self.encoder = resnets[num_layers]()

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(
            self.encoder.layer1(self.encoder.maxpool(self.features[-1]))
        )
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
