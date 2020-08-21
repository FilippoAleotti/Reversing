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
import cv2


def cv_read(path):
    """Read an image using opencv.
    Args:
        path: path to image

    Returns:
        numpy array containing the RGB image
    """
    image = cv2.imread(path)
    image = bgr_to_rgb(image)
    return image


def bgr_to_rgb(image):
    """Convert to RGB color space an image or a list of BGR images
    Args:
        image: list of images or a single image

    Returns:
        an image if a single image is given, a list of images otherwise
    """

    def _to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if isinstance(image, (list,)):
        return [_to_rgb(img) for img in image]
    else:
        return _to_rgb(image)


def image_loader(path, method=None):
    """Read an image and return it
    Args:
        path: image path
        method: function. If not None, use this to read the image
    """
    extension = os.path.splitext(path)[1].replace(".", "").lower()
    function_to_apply = {
        "png": cv_read,
        "jpeg": cv_read,
        "jpg": cv_read,
    }
    try:
        if method is not None:
            image = method(path)
        else:
            image = function_to_apply[extension](path)
    except:
        print("Cannot open image: " + path)
        image = None
    return image
