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
"""
Evaluate your depth maps on ETH3D dataset
"""

from __future__ import division
import os
import cv2
import argparse
from terminaltables import AsciiTable

parser = argparse.ArgumentParser(description="ETH3D test")
parser.add_argument("--gt", type=str, help="path to dataset", required=True)
parser.add_argument(
    "--predictions", type=str, help="path to predictions saved as 16bit", required=True
)
args = parser.parse_args()

import re
import numpy as np


def load_pfm(filename):
    """
    Read a pfm flow map
    From https://github.com/JiaRenChang/PSMNet/blob/master/dataloader/readpfm.py
    """
    color = None
    width = None
    height = None
    scale = None
    endian = None

    with open(filename, "r", encoding="latin-1") as f:
        header = f.readline().rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = "<"
            scale = -scale
        else:
            endian = ">"  # big-endian

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width, 1)

    return np.flipud(np.reshape(data, shape))


def test():
    """Run test on ETH3D.
    """
    data_path = args.gt
    if data_path[-1] == "/":
        data_path = data_path[:-1]

    bad_pixels = [0.0, 0.0, 0.0]  # bad1, bad2, bad3
    epe = 0.0
    counter = 0

    for subdir, _, files in os.walk(data_path):
        for file in files:
            gt = os.path.join(subdir, file)
            if not "pfm" in gt:
                continue
            seq = os.path.dirname(gt.replace(data_path + "/", ""))
            pred_name = os.path.join(args.predictions, seq + ".png")
            gt = load_pfm(gt)
            if not os.path.exists(pred_name):
                raise ValueError("Image {} not found".format(pred_name))
            pred = cv2.imread(pred_name, -1) / 256.0

            hp, wp = pred.shape
            hg, wg, _ = gt.shape
            factor_h = hg // hp
            factor_w = wg // wp
            if factor_h != factor_w:
                raise ValueError("Not expected shapes for image {}".format(pred_name))
            if factor_h != 1.0:
                print("Rescaling predictions by " + str(factor_h))
                pred = (
                    cv2.resize(
                        pred,
                        (wp * factor_w, hp * factor_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    * factor_w
                )

            pred = np.expand_dims(pred, -1)
            mask = gt < 5000
            scalar = 1.0

            for i in range(len(bad_pixels)):
                disp_diff = np.abs(gt[mask] - pred[mask])
                bad_pixels[i] = bad_pixels[i] + scalar * (
                    (disp_diff >= i + 1).sum() / mask.sum()
                )
            img_epe = disp_diff.mean()
            epe = epe + scalar * img_epe
            counter += scalar
    if counter == 0:
        raise ValueError("No image found")
    table_data = [
        ["BAD 1", "BAD 2", "EPE"],
        [
            "{:2f}".format(100 * bad_pixels[0] / counter),
            "{:2f}".format(100 * bad_pixels[1] / counter),
            "{:2f}".format(epe / counter),
        ],
    ]
    table = AsciiTable(table_data)
    print(table.table)


if __name__ == "__main__":
    test()
