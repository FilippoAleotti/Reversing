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
import os
import torch
import torch.nn as nn
import torch.nn.parallel
from utils import general
import options
import test
import train
from models import network_factory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

configuration = options.Option()
configuration.parse_general_params()
general_params = configuration.general_params
torch.manual_seed(general_params["seed"])

if general_params["cuda"]:
    torch.cuda.manual_seed(general_params["seed"])

if __name__ == "__main__":
    net_params = {
        "maxdisp": general_params["maxdisp"],
    }
    model = network_factory.get_network(general_params["model"])(net_params)
    model = nn.DataParallel(model)
    if general_params["cuda"]:
        print("=> selected gpu(s) with ids {}".format(*general_params["gpu_ids"]))
        model.cuda()
    print(
        "=> Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )

    if general_params["mode"] == "train":
        train.run_train(model, configuration)
    else:
        test.run_test(model, configuration)
