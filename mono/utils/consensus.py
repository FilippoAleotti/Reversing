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
import numpy as np
import cv2
import shutil


def create_dir(path):
    """Create a dir if not exists
    Args:
        path: path to new directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


def consensus_mechanism(
    name, multiple_predictions_folder, destination, number_hypothesis
):
    """Apply consensus mechanism
    Args:
        name: RGB image name
        multiple_predictions_folder: folder that contains multiple hypothesis
        destination: where the filtered map will be saved
        number_hypothesis: number of expected hypothesis
    """
    files = os.listdir(multiple_predictions_folder)

    for counter, disp_i in enumerate(files):
        disp = cv2.imread(multiple_predictions_folder + "/" + disp_i, -1) / 256.0
        if counter == 0:
            mean = np.zeros(disp.shape, dtype=np.float32)

        mean += disp

    iterations = len(files)
    if iterations != number_hypothesis:
        raise ValueError(
            "Expected {} predictions in {} folder, but found {}".format(
                number_hypothesis, multiple_predictions_folder, iterations
            )
        )
    mean = mean / iterations

    for counter, disparity_path in enumerate(files):
        disp = (
            cv2.imread(multiple_predictions_folder + "/" + disparity_path, -1) / 256.0
        )

        if counter == 0:
            confidence = np.zeros(disp.shape, dtype=np.float32)

        confidence += np.square(disp - mean)

    confidence = confidence / iterations
    final_disp = mean * (confidence < 3.0)
    name = os.path.splitext(name)[0]
    name = name + ".png"
    disp_dest = destination + "/consensus_filtered_proxies/" + name

    create_dir(os.path.dirname(disp_dest))
    cv2.imwrite(disp_dest, (final_disp * 256.0).astype(np.uint16))
    shutil.rmtree(multiple_predictions_folder)
