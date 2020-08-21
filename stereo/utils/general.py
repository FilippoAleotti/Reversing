import numpy as np
import os
import json
import cv2
from tensorboard import program
from utils.KITTI_colormap import kitti_colormap


def parse_gpu_ids(gpu_str):
    """Parse gpu ids from configuration.
    Args:
        gpu_str: string with gpu indexes
    
    Return:
        a list where each element is the gpu index as int
    """
    gpus = []
    if gpu_str:
        parsed_gpus = gpu_str.split(",")
        if len(gpus) != 1 or int(gpus[0]) >= 0:
            return [int(p) for p in parsed_gpus]
    return gpus


def create_dir(directory):
    """Create directory if not already exists
    Args:
        directory: path to the directory to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_kitti_disp(destination, disparity):
    """Save a disparity map as expected by Kitti Sceneflow Benchmark
    """
    dirname = os.path.dirname(destination)
    create_dir(dirname)
    cv2.imwrite(destination, (disparity * 256).astype(np.uint16))


def save_color(destination, disparity, cmap="magma", params=None):
    """Save a predicted disparity as colormap
    Args:
        destination: path to the image that will be saved
        disparity: predicted disparity to save
        cmap: colormap to apply
        params: dictionary with configuration

    Return:
        this method save the disparity in a png image placed at destination
    """
    dirname = os.path.dirname(destination)
    create_dir(dirname)
    colormap_func = get_cmap(cmap)
    if colormap_func is None:
        colored = disparity
    else:
        if cmap == "kitti":
            maxval = -1
            if params != None and "maxval" in params.keys():
                maxval = params["maxval"]
            colored = colormap_func(disparity, maxval=maxval)
        else:
            colored = cv2.applyColorMap(np.uint8(disparity), colormap_func)
    cv2.imwrite(destination, colored)


def run_tensorboard(logdir, port=6006):
    """Enable tensorboard logging
    Args:
        logdir: path to summary folder
        port: tensorboard port. Int, default 6006

    Return:
        this method creates a tensorboard daemon process, it will be stopped at the end of the training
    """
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", logdir, "--bind_all", "--port", str(port)])
    url = tb.launch()


def parse_weights(weights):
    """Parse loss weights from configuration setting.
    Args:
        weights: weights in configuration
    
    Return:
        a list where each element is the weight as float
    """
    weights = weights.split(",")
    weights = [float(w) for w in weights]
    return weights


def get_cmap(cmap):
    """Get the right function to apply a desired colormap.
    Args:
        cmap: colormap to apply

    Return:
        colormap function
    """
    AVAILABLE_CMAPS = {
        "magma": cv2.COLORMAP_MAGMA,
        "jet": cv2.COLORMAP_JET,
        "gray": None,
        "kitti": kitti_colormap,
    }
    assert cmap in AVAILABLE_CMAPS.keys()
    return AVAILABLE_CMAPS[cmap]


def write_cfg(params, dest):
    """ Serialize training configuration in a file"""
    if not dest.endswith(".json"):
        dest += ".json"
    with open(dest, "w") as f:
        f.write(json.dumps(params))
