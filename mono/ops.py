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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def correlation_map(x, y, max_disp, stride=1, name="corr"):
    """Get correlation volume between x and y
    Args:
        x: tensor with shape BxHxWxC
        y: tensor with shape BxHxWxC
        max_disp: maximum correlation shift
        stride: skip in consecutive correlation shifts
        name: name of operation. Default is corr
    Return:
        a correlation volume
    """
    with tf.variable_scope(name):
        corr_tensors = []
        y_shape = tf.shape(y)
        y_feature = tf.pad(y, [[0, 0], [0, 0], [max_disp, max_disp], [0, 0]])
        for i in range(-max_disp, max_disp + 1, stride):
            shifted = tf.slice(
                y_feature, [0, 0, i + max_disp, 0], [-1, y_shape[1], y_shape[2], -1]
            )
            corr_tensors.append(tf.reduce_mean(shifted * x, axis=-1, keepdims=True))

        result = tf.concat(corr_tensors, axis=-1)
        return result


def conv2d(x, kernel_shape, strides=1, relu=True, padding="SAME"):
    """Get convolution 2D filter
    Args:
        x: input features
        kernel_shape: shape of kernel
        strides: convolution strides. Default is 1
        relu: boolean. Apply relu activation as last operation
        padding: type of padding. Choices are SAME or VALID. Default is SAME
    Return:
        Convolution activations
    """
    W = tf.get_variable(
        "weights",
        kernel_shape,
        initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
    )
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    b = tf.get_variable(
        "biases", kernel_shape[3], initializer=tf.constant_initializer(0.0)
    )
    with tf.name_scope("conv"):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        if relu:
            x = tf.nn.relu(x)

    return x


def conv2d_transpose(x, kernel_shape, strides=1, relu=True):
    """Get 2D transposed convolution filter
    Args:
        x: input features
        kernel_shape: shape of kernel
        strides: convolution strides. Default is 1
        relu: boolean. Apply relu activation as last operation
    Return:
        Transposed convolution activations
    """
    W = tf.get_variable(
        "weights_transpose",
        kernel_shape,
        initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
    )
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    b = tf.get_variable(
        "biases_transpose", kernel_shape[2], initializer=tf.constant_initializer(0.0)
    )
    output_shape = [
        x.get_shape()[0].value,
        x.get_shape()[1].value * strides,
        x.get_shape()[2].value * strides,
        kernel_shape[2],
    ]
    with tf.name_scope("deconv"):
        x = tf.nn.conv2d_transpose(
            x, W, output_shape, strides=[1, strides, strides, 1], padding="SAME"
        )
        x = tf.nn.bias_add(x, b)
        if relu:
            x = tf.nn.relu(x)
    return x

