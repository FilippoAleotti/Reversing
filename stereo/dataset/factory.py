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

from dataset import KITTI, DS, MIDDLEBURY, ETH3D

# NOTE: at training time, both KITTI and DS load the images at the same way
TRAINING_DATASET_FACTORY = {
    "DS": KITTI.load_from_file,
    "KITTI": KITTI.load_from_file,
}

TESTING_DATASET_FACTORY = {
    "DS": DS.load_from_file_test,
    "KITTI": KITTI.load_from_folder,
    "ETH3D": ETH3D.load_from_file,
    "MIDDLEBURY": MIDDLEBURY.load_from_file,
}

TESTING_FILENAMES = {
    "DS": "filenames/drivingstereo-test.txt",
    "ETH3D": "filenames/eth3d.txt",
    "MIDDLEBURY": "filenames/middlebury.txt",
}


def get_dataset_train(dataset_name):
    """Return the desired dataset for training.
    Args:
        dataset_name: name of dataset to load
    Return:
        Desired dataloader
    """
    if dataset_name not in TRAINING_DATASET_FACTORY.keys():
        raise ValueError("Dataset not available!")
    return TRAINING_DATASET_FACTORY[dataset_name]


def get_dataset_test(dataset_name):
    """Return the desired dataset for testing.
    Args:
        dataset_name: name of dataset to load
    Return:
        Desired dataloader
    """
    if dataset_name not in TESTING_DATASET_FACTORY.keys():
        raise ValueError("Dataset not available!")
    return TESTING_DATASET_FACTORY[dataset_name]


def get_test_file_path(dataset_name, path_to_folder):
    """Return the correct filename file or folder to load for testing.
    If the dataset has to load a filename_file, then return the path to
    the filename_file. Otherwise, return the path to the folder with images
    Args:
        dataset_name: name of dataset to load
        path_to_folder: default path to folder if dataset has to load a folder
    Return:
        path to folder with images or to a filename file
    """
    if dataset_name in TESTING_FILENAMES.keys():
        return TESTING_FILENAMES[dataset_name]
    else:
        return path_to_folder
