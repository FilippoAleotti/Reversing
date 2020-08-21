import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np


def load_samples(line, rgb_ext, proxy_ext):
    """Load samples in line.
    Args:
        line: n-th line of filenames_file
        rgb_ext: extension of RGB image
        proxy_ext: extension of proxy file

    Return:
        left: np array containing the left image
        right: np array containing the right image
        proxy_left: np array containing the proxy left image
        proxy_right: np array containing the proxy right image
    """
    left, right = line.replace("\n", "").split(" ")
    proxy_left = left
    proxy_right = right
    left, right = [set_extension(x, rgb_ext) for x in [left, right]]
    proxy_left, proxy_right = [
        set_extension(x, proxy_ext) for x in [proxy_left, proxy_right]
    ]
    return left, right, proxy_left, proxy_right


def set_extension(path, ext):
    """Set the extension of file.
    Given the path to a file and the desired extension, set the extension of the
    file to the given one.

    Args:
        path: path to a file
        ext: desired extension

    Return:
        path with the correct extension
    """
    ext = ext.replace(".", "")
    current_ext = path.split(".")
    if len(current_ext) == 2:
        path = path.replace(current_ext[1], ext)
    else:
        path = path + "." + ext
    return path


def load_from_file(filepath, rgb_ext, proxy_ext):
    """Load samples from a filename file.
    Args:
        filepath: path to filename file
        rgb_ext: extension of rgb images
        proxy_ext: extension of proxy images

    Return:
        left_train: list of paths to left images
        right_train: list of paths to right images
        proxy_left: list of paths to proxy left images
        proxy_right: list of paths to proxy right images
    """
    left_train = []
    right_train = []
    proxy_left = []
    proxy_right = []

    with open(filepath) as f:
        lines = f.readlines()

    for l in lines:
        left, right, pl, pr = load_samples(l, rgb_ext, proxy_ext)
        left_train.append(left)
        right_train.append(right)
        proxy_left.append(pl)
        proxy_right.append(pr)

    return left_train, right_train, proxy_left, proxy_right


def load_from_folder(folder):
    """Load samples from a folder.
    Args:
        folder: path to folder
        is_2015: True if folder is KITTI 2015, False if folder is KITTI 2012

    Return:
        left: list of paths to left images
        right: list of paths to right images
    """
    left_fold = "image_2/"
    right_fold = "image_3/"
    num_samples = 200

    left = [
        os.path.join(folder, left_fold, str(i).zfill(6) + "_10.png")
        for i in range(num_samples)
    ]
    right = [
        os.path.join(folder, right_fold, str(i).zfill(6) + "_10.png")
        for i in range(num_samples)
    ]
    return left, right
