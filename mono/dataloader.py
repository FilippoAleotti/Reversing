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
from collections import namedtuple


dataloader_parameters = namedtuple(
    "dataloader_parameters",
    "patch_height, patch_width, "
    "height, width, "
    "batch_size, "
    "is_right, "
    "num_threads, ",
)


def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


class Dataloader(object):
    def __init__(
        self,
        data_path_image,
        data_path_proxy,
        filenames_file,
        dataset,
        is_training,
        image_path,
        params,
    ):
        self.data_path_image = data_path_image
        self.data_path_proxy = data_path_proxy
        self.params = params
        self.dataset = dataset
        self.image_path = image_path
        self.is_training = is_training
        self.left_image_batch = None
        self.right_image_batch = None
        self.proxy_left_batch = None
        self.proxy_right_batch = None
        self.occlusion_handler_batch = None

        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)
        self.split_line = tf.string_split([line]).values

        if is_training:
            self.left_image_path = tf.string_join(
                [self.data_path_image, self.split_line[0]]
            )
            self.right_image_path = tf.string_join(
                [self.data_path_image, self.split_line[1]]
            )
            self.proxy_left_path = tf.string_join(
                [self.data_path_proxy, self.split_line[2]]
            )
            self.proxy_right_path = tf.string_join(
                [self.data_path_proxy, self.split_line[3]]
            )

            left_image_o = self.read_image(self.left_image_path)
            right_image_o = self.read_image(self.right_image_path)
            proxy_left_o = self.read_proxy(self.proxy_left_path)
            proxy_right_o = self.read_proxy(self.proxy_right_path)

            do_flip = tf.random_uniform([], 0, 1)
            left_image = tf.cond(
                do_flip > 0.5,
                lambda: tf.image.flip_left_right(right_image_o),
                lambda: left_image_o,
            )
            right_image = tf.cond(
                do_flip > 0.5,
                lambda: tf.image.flip_left_right(left_image_o),
                lambda: right_image_o,
            )
            proxy_left = tf.cond(
                do_flip > 0.5,
                lambda: tf.image.flip_left_right(proxy_right_o),
                lambda: proxy_left_o,
            )
            proxy_right = tf.cond(
                do_flip > 0.5,
                lambda: tf.image.flip_left_right(proxy_left_o),
                lambda: proxy_right_o,
            )

            do_occlusion = tf.random_uniform([], 0, 1)
            occlusion_handler = tf.logical_and(do_occlusion < 0.5, do_flip < 0.5)
            left_image = tf.cond(
                occlusion_handler, lambda: right_image_o, lambda: left_image_o
            )
            right_image = tf.cond(
                occlusion_handler,
                lambda: tf.zeros_like(right_image_o),
                lambda: right_image_o,
            )
            proxy_left = tf.cond(
                occlusion_handler, lambda: proxy_right_o, lambda: proxy_left_o
            )
            proxy_right = tf.cond(
                occlusion_handler,
                lambda: tf.zeros_like(proxy_right_o),
                lambda: proxy_right_o,
            )

            # randomly augment images
            do_augment = tf.random_uniform([], 0, 1)
            left_image, right_image = tf.cond(
                do_augment > 0.5,
                lambda: self.augment_image_pair(left_image, right_image),
                lambda: (left_image, right_image),
            )

            left_image.set_shape([None, None, 3])
            right_image.set_shape([None, None, 3])
            proxy_left.set_shape([None, None, 1])
            proxy_right.set_shape([None, None, 1])
            for x in range(3):
                occlusion_handler = tf.expand_dims(occlusion_handler, axis=0)

            crops = tf.random_crop(
                tf.concat([left_image, right_image, proxy_left, proxy_right], -1),
                [self.params.patch_height, self.params.patch_width, 8],
            )
            left_image, right_image, proxy_left, proxy_right = tf.split(
                crops, [3, 3, 1, 1], axis=2
            )

            min_after_dequeue = 32
            capacity = min_after_dequeue + self.params.batch_size

            (
                self.left_image_batch,
                self.right_image_batch,
                self.proxy_left_batch,
                self.proxy_right_batch,
                self.occlusion_handler_batch,
            ) = tf.train.shuffle_batch(
                [left_image, right_image, proxy_left, proxy_right, occlusion_handler],
                self.params.batch_size,
                capacity,
                min_after_dequeue,
                self.params.num_threads,
            )
        else:
            index = 1 if params.is_right else 0

            self.left_image_path = tf.string_join(
                [self.data_path_image, self.split_line[index]]
            )

            self.right_image_path = tf.string_join(
                [self.data_path_image, self.split_line[index]]
            )
            self.proxy_left_path = tf.string_join(
                [self.data_path_proxy, self.split_line[2 + index]]
            )
            self.proxy_right_path = tf.string_join(
                [self.data_path_proxy, self.split_line[2 + index]]
            )

            left_image_o = tf.image.flip_left_right(
                self.read_image(self.left_image_path)
            )
            right_image_o = tf.image.flip_left_right(
                self.read_image(self.right_image_path)
            )
            proxy_left_o = tf.image.flip_left_right(
                self.read_proxy(self.proxy_left_path)
            )
            proxy_right_o = tf.image.flip_left_right(
                self.read_proxy(self.proxy_right_path)
            )

            left_image_o, right_image_o = self.augment_image_pair(
                left_image_o, right_image_o
            )

            self.name = self.split_line[index]

            self.left_image_batch = tf.stack(
                [left_image_o, tf.image.flip_left_right(left_image_o)], 0
            )
            self.right_image_batch = tf.stack(
                [right_image_o, tf.image.flip_left_right(right_image_o)], 0
            )
            self.proxy_left_batch = tf.stack(
                [proxy_left_o, tf.image.flip_left_right(proxy_left_o)], 0
            )
            self.proxy_right_batch = tf.stack(
                [proxy_right_o, tf.image.flip_left_right(proxy_right_o)], 0
            )

            self.left_image_batch.set_shape([2, None, None, 3])
            self.right_image_batch.set_shape([2, None, None, 3])
            self.proxy_left_batch.set_shape([2, None, None, 1])
            self.proxy_right_batch.set_shape([2, None, None, 1])

    def augment_image_pair(self, left_image, right_image):
        """Augment images
        From https://github.com/mrharicot/monodepth
        Args:
            left_image: left image, with shape HxWx3
            right_image: right_image, with shape HxWx3
        Return:
            left_image_aug, right_image_aug: list of augmented images
        """
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def read_image(self, image_path):
        """Read an image from the file system
        Args:
            image_path: path to image
        Return:
            tensor with shape HxWxC, where H and W are the height and width
            after the resize
        """
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, "jpg")

        image = tf.cond(
            file_cond,
            lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
            lambda: tf.image.decode_png(tf.read_file(image_path)),
        )

        self.image_w = tf.cast(tf.shape(image)[1], tf.float32)
        self.image_h = tf.cast(tf.shape(image)[0], tf.float32)

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(
            image,
            [self.params.height, self.params.width],
            tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        return image

    def read_proxy(self, image_path):
        """Read a proxy map from the file system
        Args:
            image_path: path to image
        Return:
            tensor with shape HxWx1, where H and W are the height and width
            after the resize
        """
        image = tf.image.decode_png(tf.read_file(image_path), dtype=tf.uint8)
        image = tf.cast(image, tf.float32)

        image_w = tf.cast(tf.shape(image)[1], tf.float32)
        image = tf.image.resize_images(
            image,
            [self.params.height, self.params.width],
            tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        image = image * (self.params.width / image_w)
        return image
