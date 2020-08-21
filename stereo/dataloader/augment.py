"""
Adapted from https://github.com/ClementPinard/FlowNetPytorch/blob/master/flow_transforms.py
By Clement Pinard
License: MIT
"""

from __future__ import division

import torch
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage
from torchvision import transforms
import itertools
import cv2

"""Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays"""


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input, target = t(input, target)
        return input, target


class ComposeTransformation(object):
    """ Composes several transforms together.
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input):
        for t in self.co_transforms:
            input = t(input)
        return input


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert isinstance(array, np.ndarray)
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()


class Lambda(object):
    """Applies a lambda as a transform"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, input, target):
        return self.lambd(input, target)


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, targets):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs, targets

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs = [inp[y1 : y1 + th, x1 : x1 + tw, :] for inp in inputs]
        if targets is not None:
            targets = [t[y1 : y1 + th, x1 : x1 + tw, :] for t in targets]
        return inputs, targets


class FlipLeftRight(object):
    """Change left samples with right ones and FLIP them
    """

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, inputs, targets):
        prob = np.random.uniform(0, 1, 1)
        if prob > self.prob:
            return inputs, targets

        left = np.ascontiguousarray(np.fliplr(inputs[1]), dtype=np.float32)
        right = np.ascontiguousarray(np.fliplr(inputs[0]), dtype=np.float32)

        target_left = np.ascontiguousarray(np.fliplr(targets[1]), dtype=np.float32)
        target_right = np.ascontiguousarray(np.fliplr(targets[0]), dtype=np.float32)

        return [left, right], [target_left, target_right]


class ColorAugmentation(object):
    """Based on the color augmentation of https://github.com/ClubAI/MonoDepth-PyTorchs
    """

    def __init__(self):
        self.gamma_low = 0.8
        self.gamma_high = 1.2
        self.brightness_low = 0.5
        self.brightness_high = 2.0
        self.color_low = 0.8
        self.color_high = 1.2

    def __call__(self, inputs, targets):
        left, right = inputs
        p = np.random.uniform(0, 1, 1)

        if p > 0.5:
            # randomly shift gamma
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            left_aug = left ** random_gamma
            right_aug = right ** random_gamma

            # randomly shift brightness
            random_brightness = np.random.uniform(
                self.brightness_low, self.brightness_high
            )
            left_aug = left_aug * random_brightness
            right_aug = right_aug * random_brightness

            # randomly shift color
            random_colors = np.random.uniform(self.color_low, self.color_high, 3)
            for i in range(3):
                left_aug[i, :, :] *= random_colors[i]
                right_aug[i, :, :] *= random_colors[i]

            # saturate
            left_aug = np.clip(left_aug, 0, 255)
            right_aug = np.clip(right_aug, 0, 255)

            return [left_aug, right_aug], targets
        else:
            return [left, right], targets


class ImagePadder(object):
    """Pad an image to a selected shape given a method.
        Method has to be in [reflective, zero, replicate].
        Final shape is [h,w]
    """

    def _pad(self, x):
        h, w, _ = x.shape
        assert self.final_shape[0] >= h and self.final_shape[1] >= w
        missing_h = self.final_shape[0] - h
        missing_w = self.final_shape[1] - w

        missing_top = np.floor(missing_h / 2).astype(np.int32)
        missing_bottom = missing_h - missing_top

        missing_left = np.floor(missing_w / 2).astype(np.int32)
        missing_right = missing_w - missing_left

        if self.method == "reflective":
            return cv2.copyMakeBorder(
                x,
                top=missing_top,
                bottom=missing_bottom,
                left=missing_left,
                right=missing_right,
                borderType=cv2.BORDER_REFLECT,
            )
        elif self.method == "zero":
            return cv2.copyMakeBorder(
                x,
                top=missing_top,
                bottom=missing_bottom,
                left=missing_left,
                right=missing_right,
                borderType=cv2.BORDER_CONSTANT,
                value=0,
            )
        elif self.method == "replicate":
            return cv2.copyMakeBorder(
                x,
                top=missing_top,
                bottom=missing_bottom,
                left=missing_left,
                right=missing_right,
                borderType=cv2.BORDER_REPLICATE,
            )
        else:
            raise ValueError("not a valid padding method")

    def __init__(self, final_shape, method="reflective"):
        assert method in ["reflective", "zero", "replicate"]
        self.final_shape = final_shape
        self.method = method

    def __call__(self, inputs):
        padded_inputs = []

        for _, x in enumerate(inputs):
            padded_inputs.append(self._pad(x))
        return padded_inputs
