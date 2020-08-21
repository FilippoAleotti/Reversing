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

from models.iresnet import network as iresnet
from models.psm import network as psm
from models.monodepth2 import network as monodepth2
from models.gwcnet import network as gwcnet

NETWORK_FACTORY = {
    "psm": psm.PSMNet,
    "iresnet": iresnet.iResNet,
    "stereodepth": monodepth2.MonoDepth2,
    "gwcnet": gwcnet.GwcNet,
}


def get_network(model):
    AVAILABLE_NETWORKS = NETWORK_FACTORY.keys()
    assert model in AVAILABLE_NETWORKS
    return NETWORK_FACTORY[model]
