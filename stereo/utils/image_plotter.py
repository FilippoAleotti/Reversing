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
import cv2
from utils.tensor import to_tensor, to_regular_format
import matplotlib
import matplotlib.cm


class Plotter(object):
    def __init__(self, writer):
        self.writer = writer

    def _normalize(self, x):
        """Normalize between 0.0 and 255.0 the input.
        Args:
            x: input tensor
        Return:
            normalized input, rescaled in the range [0, 255]
        """
        min_value = x.min()
        max_value = x.max()
        return 255.0 * (x - min_value) / (max_value - min_value)

    def plot_summary_images(self, summary_images, step):
        """Add to writer the images to be plotted.
        Args:
            summary_images: list of summary_images to plot
            step: current step
        """
        for index, sum_img in enumerate(summary_images):
            image_params = sum_img.settings
            image_name = image_params["name"] + "/frame_" + str(index)
            img = self._transform_image(sum_img.image, image_params)
            self.writer.add_image(image_name, img, step)

    def _transform_image(self, tensor_image, settings):
        """Apply image transformations.
        Args:
            tensor_image: input image as a tensor, with shape BxCxHxW
            settings: dictionary with configuration
        Return:
            tensor image after transformations
        """

        if settings["disp"]:

            settings["bgr2rgb"] = True
            norm = lambda img, params: self._normalize(img)
            func = lambda img, params: cv2.applyColorMap(
                np.uint8(img), cv2.COLORMAP_MAGMA
            )
            tensor_image = self._map(tensor_image, [norm, func], settings)
        if settings["bgr2rgb"]:
            func = lambda img, params: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor_image = self._map(tensor_image, [func], settings)

        return tensor_image

    def _map(self, tensor, functions, params):
        """Apply a given function to a tensor, transforming it into numpy array
        Args:
            tensor: input tensor with shape BxCxHxW
            functions: list of functions to apply
            params: dictionary with configuration
        Return:
            apply all the transformation functions to tensor and return the final result
        """
        numpy_image = to_regular_format(tensor)
        numpy_image = numpy_image.cpu().detach().numpy()
        img = numpy_image
        for func in functions:
            trans_img = func(img, params)
            img = trans_img
        return to_tensor(trans_img)

    def prepare_summary(self, name, data, disp=False, max_el=None):
        """Prepare printable image
        Args:
            name : name to visualize in tensorboard
            data : tensor to visualize
            max_el: max number of frames to plot. This param is valid only if != None and < batch size
            disp: data is a disparity

        Return:
            list of summary images
        """
        summary_images = []

        batch = data.shape[0]
        number_elements = batch
        if max_el is not None and max_el < batch:
            number_elements = max_el
        for i in range(number_elements):
            summary_image = SummaryImage(data[i, :, :, :], name=name)
            summary_image.change_settings({"disp": disp})
            summary_images.append(summary_image)
        return summary_images


class SummaryImage(object):
    """Objects that contain an image that can be printed using TensorBoard
    """

    def __init__(self, image, name):
        self.image = image
        self.settings = {
            "name": name,
            "disp": False,
            "bgr2rgb": False,
            "final_shape": None,
        }

    def change_setting(self, key, value):
        """Change the setting
        Args:
            key: key value to change
            value: new value for the provided key
        Return:
            change the dictionary with the new value
        """
        self.settings[key] = value

    def change_settings(self, settings):
        """Change the setting
        Args:
            setting: new configuration dictionary
        Return:
            change the dictionary with the new values
        """
        if settings is None:
            return
        for k, v in settings.items():
            self.settings[k] = v
