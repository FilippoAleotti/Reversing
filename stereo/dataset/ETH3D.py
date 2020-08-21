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

import os


def load_from_file(filename):
    """Load samples from a filename file.
    Args:
        filename: path to filename file

    Return:
        left: list of paths to left images
        right: list of paths to right images
    """
    with open(filename) as f:
        lines = f.readlines()

    left_img = "im0.png"
    right_img = "im1.png"

    left = [os.path.join(l.strip(), left_img) for l in lines]
    right = [os.path.join(l.strip(), right_img) for l in lines]
    return left, right
