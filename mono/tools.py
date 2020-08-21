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

import numpy as np
import tensorflow as tf
import cv2
import matplotlib


def colorize(value, vmin=None, vmax=None, cmap=None):
    """Apply a colormap to a tensor
    Args:
        value: input tensor with shape BxHxWxC
        vmin: min value. Default is None
        vmax: max value. Default is None
        cmap: matplotlib colormap to apply. Default is None -> gray colormap
    Return:
        a tensor with the same shape of the input, with the result of the colormap
    """
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    value = tf.squeeze(value)
    indices = tf.to_int32(tf.round(value * 255))
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else "gray")
    colors = tf.constant(cm.colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
    return value


def count_text_lines(txt_file):
    """Get number of lines in a text file
    Args:
        txt_file: txt file to count
    Return:
        int, the number of lines in the input file
    """
    f = open(txt_file, "r")
    lines = f.readlines()
    f.close()
    return len(lines)

