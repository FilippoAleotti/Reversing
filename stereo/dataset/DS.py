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

from dataset import KITTI


def load_from_file_test(filepath, rgb_ext):
    """Load testing images.
    Args:
        filepath: path to filename file
        rgb_ext: extension of RGB images

    Return:
        left_test: list of paths to left images
        right_test: list of paths to right images
    """
    left_test = []
    right_test = []

    with open(filepath) as f:
        lines = f.readlines()

    for line in lines:
        left, right = _load_samples_test(line, rgb_ext)
        left_test.append(left)
        right_test.append(right)

    return left_test, right_test


def _load_samples_test(line, rgb_ext):
    """Load samples in line.
    Args:
        line: line to load
        rgb_ext: extension of RGB image

    Return:
        left: list of paths to left images
        right: list of paths to right images
    """
    left, right = line.replace("\n", "").split(" ")
    left, right = [KITTI.set_extension(x, rgb_ext) for x in [left, right]]
    return left, right
