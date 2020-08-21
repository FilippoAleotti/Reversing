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

from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from models.iresnet.submodule import (
    stem_block,
    disparity_estimation,
    disparity_refinement,
)


class iResNet(nn.Module):
    def __init__(self, params):
        super(iResNet, self).__init__()
        self.params = params
        self.stem_block = stem_block()
        self.disparity_estimation = disparity_estimation()
        self.disparity_refinement = disparity_refinement()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        """Run the network.
        Args:
            left: left image of the stereo pair. Tensor with shape BxHxWx3
            right: right image of the stereo pair. Tensor with shape BxHxWx3
        Return:
            A tensor with shape BxHxWx1 containing disparity values
        """
        conv1a, conv2a, up_1a2a = self.stem_block(left)
        conv1b, conv2b, up_1b2b = self.stem_block(right)

        # cost volume's dimensions
        batch_size = conv2a.size()[0]
        num_disp = 40
        height = conv2a.size()[2]
        width = conv2a.size()[3]

        # correlation layer
        corr1d = Variable(
            torch.FloatTensor(batch_size, num_disp * 2 + 1, height, width).zero_()
        ).cuda()
        pad_conv2b = F.pad(conv2b, (num_disp, num_disp), "constant", 0)

        for i in range(-num_disp, num_disp + 1):
            if i != 0:
                corr1d[:, i + num_disp, :, :] = torch.mean(
                    conv2a[:, :, :, :]
                    * pad_conv2b[:, :, :, i + num_disp : i + num_disp + width],
                    dim=1,
                    keepdim=False,
                )
            else:
                corr1d[:, num_disp, :, :] = torch.mean(
                    conv2a[:, :, :, :]
                    * pad_conv2b[:, :, :, num_disp : num_disp + width],
                    dim=1,
                    keepdim=False,
                )
        corr1d = corr1d.contiguous()

        disp0, disp1, disp2, _, _, _, _ = self.disparity_estimation(
            conv1a, up_1a2a, conv2a, corr1d
        )
        r_res0, r_res1, r_res2 = self.disparity_refinement(
            up_1a2a, up_1b2b, conv1a, conv1b, disp0
        )

        if self.training:
            upsampled_predictions = []
            for x in [disp1, r_res1, disp2, r_res2]:
                up = F.interpolate(
                    x,
                    [left.size()[2], left.size()[3]],
                    mode="bilinear",
                    align_corners=True,
                )
                upsampled_predictions.append(up)
            upsampled_predictions += [disp0, r_res0]

            return upsampled_predictions
        else:
            return r_res0
