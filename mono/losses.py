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

import tensorflow as tf


def SSIM(x, y):
    """Get Structural Similarity Index
    From https://github.com/mrharicot/monodepth
    Args:
        x: tensor with shape BxHxWxC
        y: tensor with shape BxHxWxC
    Return:
        SSIM between x and y
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], "VALID")
    mu_y = tf.nn.avg_pool(y, [1, 3, 3, 1], [1, 1, 1, 1], "VALID")

    sigma_x = tf.nn.avg_pool(x ** 2, [1, 3, 3, 1], [1, 1, 1, 1], "VALID") - mu_x ** 2
    sigma_y = tf.nn.avg_pool(y ** 2, [1, 3, 3, 1], [1, 1, 1, 1], "VALID") - mu_y ** 2
    sigma_xy = tf.nn.avg_pool(x * y, [1, 3, 3, 1], [1, 1, 1, 1], "VALID") - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


def berhu_loss(labels, predictions, scope=None):
    """BerHu loss
    Args:
        labels: ground truth values
        predictions: predictions of the network
        scope: optional scope of this operation
    Return:
        pixel-wise BerHu loss
    """
    with tf.name_scope(scope, "berhu_loss", (predictions, labels)):
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)

        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        abs_error = tf.abs(tf.subtract(predictions, labels), name="abs_error")

        c = 0.2 * tf.reduce_max(abs_error)
        berHu_loss = tf.where(
            abs_error <= c, abs_error, (tf.square(abs_error) + tf.square(c)) / (2 * c)
        )

        return berHu_loss


def gradient_x(img):
    """Get horizontal gradient
    From https://github.com/mrharicot/monodepth
    Args:
        img: input rgb image
    Return:
        First derivative of img along horizontal direction
    """
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    gx = tf.pad(gx, [[0, 0], [0, 0], [0, 1], [0, 0]])
    return gx


def gradient_y(img):
    """Get vertical gradient
    From https://github.com/mrharicot/monodepth
    Args:
        img: input rgb image
    Return:
        First derivative of img along vertical direction
    """
    gy = img[:, :-1, :, :] - img[:, 1:, :, :]
    gy = tf.pad(gy, [[0, 0], [0, 1], [0, 0], [0, 0]])
    return gy


def get_disparity_smoothness(d, img):
    """Measure disparity smoothness between disparity and rgb image
    From https://github.com/mrharicot/monodepth
    Args
        d: disparity value predicted by the network
        img: input rgb image
    Return:
        Edge aware smoothness term
    """
    disp_gradients_x = gradient_x(d)
    disp_gradients_y = gradient_y(d)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return smoothness_x + smoothness_y
