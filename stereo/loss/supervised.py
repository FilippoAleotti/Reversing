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
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.modules.Module):
    def __init__(self, loss_weights):
        super(Loss, self).__init__()
        self.loss_weights = loss_weights

    def _expand_dims(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        return x

    def _get_valid(self, target):
        target = torch.squeeze(target, 1)
        mask = target > 0
        mask = self._expand_dims(mask)
        target = self._expand_dims(target)
        mask.detach()
        valid = target[mask].size()[0]
        return mask, valid

    def forward(self, predictions, target):
        mask, valid = self._get_valid(target)
        if valid > 0:
            loss = [
                F.smooth_l1_loss(predictions[i][mask], target[mask])
                * self.loss_weights[i]
                for i in range(len(predictions))
            ]
            loss = torch.stack(loss)
            return torch.mean(loss)
        else:
            return 0.0
