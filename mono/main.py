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

import argparse
import tensorflow as tf
import time
import numpy as np
import os
import cv2
from model import *
from dataloader import *
from tools import *
from tqdm import tqdm
from utils import consensus
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Monocular Completion Network args")

# Arguments related to run mode
parser.add_argument("--is_training", help="train, test", action="store_true")
parser.add_argument(
    "--cpu", help="the network runs on CPU if enabled", action="store_true"
)
parser.add_argument(
    "--number_hypothesis",
    type=int,
    default=25,
    help="number of prediction to mix with the consensus mechanism",
)
parser.add_argument(
    "--right", action="store_true", help="load right images instead of left"
)
parser.add_argument(
    "--temp_folder", type=str, help="load right images instead of left", default="temp"
)

# Arguments related to training
parser.add_argument(
    "--iterations", dest="iterations", type=int, default=300000, help="# of iterations"
)
parser.add_argument(
    "--batch_size", dest="batch_size", type=int, default=2, help="# images in batch"
)
parser.add_argument(
    "--patch_width",
    dest="patch_width",
    type=int,
    default=512,
    help="# images in patches",
)
parser.add_argument(
    "--patch_height",
    dest="patch_height",
    type=int,
    default=256,
    help="# images in patches",
)
parser.add_argument(
    "--width", dest="width", type=int, default=1280, help="# image height"
)
parser.add_argument(
    "--height", dest="height", type=int, default=384, help="# image width"
)
parser.add_argument(
    "--initial_learning_rate",
    dest="initial_learning_rate",
    type=float,
    default=0.0001,
    help="initial learning rate for gradient descent",
)
parser.add_argument(
    "--learning_rate_scale_factor",
    dest="learning_rate_scale_factor",
    type=float,
    default=2.0,
    help="lr will be reduced to lr/learning_rate_scale_factor every N steps",
)
parser.add_argument(
    "--learning_rate_schedule",
    type=str,
    help="Enter the list of steps in which the learning rate will be reduced",
)
parser.add_argument(
    "--num_threads", dest="num_threads", type=int, default=4, help="num_threads"
)
parser.add_argument(
    "--scales_initial",
    dest="scales_initial",
    type=int,
    default=4,
    help="number of considered scales during the initial disparity loss computation",
)
parser.add_argument(
    "--scales_refined",
    dest="scales_refined",
    type=int,
    default=3,
    help="number of considered scales during the disparity refinement loss computation",
)
parser.add_argument(
    "--max_to_keep",
    dest="max_to_keep",
    type=int,
    default=5,
    help="indicates the maximum number of recent checkpoint files to keep",
)

# Arguments related to dataset
parser.add_argument(
    "--dataset", dest="dataset", type=str, default="kitti", help="name dataset"
)
parser.add_argument(
    "--data_path_image", type=str, default="", help="dataset path image",
)
parser.add_argument(
    "--data_path_proxy", type=str, default="", help="dataset path proxy",
)
parser.add_argument(
    "--filenames_file",
    type=str,
    default="./utils/filenames/kitti_train_files.txt",
    help="filenames_file path",
)
parser.add_argument(
    "--image_path", type=str, default="./test/example0.png", help="single image path",
)

# Arguments related to monitoring and outputs
parser.add_argument(
    "--log_directory",
    type=str,
    default="./log",
    help="directory to save checkpoints and summaries",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="",
    help="path to a specific checkpoint to load",
)
parser.add_argument(
    "--save_iter_freq",
    type=int,
    default=5000,
    help="save a model every save_iter_freq steps (does not overwrite previously saved models)",
)
parser.add_argument(
    "--model_name", dest="model_name", type=str, default="model", help="model name"
)
parser.add_argument(
    "--output_path", dest="output_path", type=str, default="./output", help="model name"
)
parser.add_argument(
    "--display_factor",
    type=float,
    default=2.0,
    help="display_factor scales the output disparity map for visualization only",
)
parser.add_argument(
    "--display_step",
    type=int,
    default=100,
    help="every display_step the training current state will be printed and the summary updated",
)
parser.add_argument(
    "--retrain",
    help="if used with checkpoint_path, will restart training from step zero",
    action="store_true",
)
parser.add_argument(
    "--input_points",
    type=float,
    help="threeshold about number of input points used at testing time",
    default=0.95,
)

# Arguments related to losses
parser.add_argument(
    "--alpha_SSIM_L1",
    type=float,
    help="weight between SSIM and L1 in the image loss",
    default=0.85,
)
parser.add_argument(
    "--alpha_image_loss", type=float, help="image loss weigth", default=1.0
)
parser.add_argument(
    "--alpha_proxy_loss", type=float, help="proxy loss weigth", default=1.0
)
parser.add_argument(
    "--alpha_smoothness_loss",
    type=float,
    help="disparity smoothness weigth",
    default=0.01,
)
args = parser.parse_args()

if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def configure_parameters():

    network_params = network_parameters(
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        alpha_SSIM_L1=args.alpha_SSIM_L1,
        alpha_image_loss=args.alpha_image_loss,
        alpha_proxy_loss=args.alpha_proxy_loss,
        alpha_smoothness_loss=args.alpha_smoothness_loss,
        scales_initial=args.scales_initial,
        scales_refined=args.scales_refined,
        display_factor=args.display_factor,
        input_points=args.input_points,
    )

    dataloader_params = dataloader_parameters(
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        is_right=args.right,
        num_threads=args.num_threads,
    )

    return network_params, dataloader_params


def configure_network(network_params, dataloader_params):
    placeholders = []
    data_path_image = args.data_path_image
    if not data_path_image.endswith("/"):
        data_path_image += "/"
    data_path_proxy = args.data_path_proxy
    if not data_path_proxy.endswith("/"):
        data_path_proxy += "/"
    dataloader = Dataloader(
        data_path_image,
        data_path_proxy,
        args.filenames_file,
        args.dataset,
        args.is_training,
        args.image_path,
        dataloader_params,
    )
    if args.is_training:
        network = Network(
            dataloader.left_image_batch,
            dataloader.right_image_batch,
            dataloader.proxy_left_batch,
            dataloader.proxy_right_batch,
            dataloader.occlusion_handler_batch,
            args.is_training,
            ["monoResMatch"],
            network_params,
        )
    else:
        left = tf.placeholder(tf.float32, shape=[2, args.height, args.width, 3])
        right = tf.placeholder(tf.float32, shape=[2, args.height, args.width, 3])
        proxy_left = tf.placeholder(tf.float32, shape=[2, args.height, args.width, 1])
        proxy_right = tf.placeholder(tf.float32, shape=[2, args.height, args.width, 1])
        placeholders = [left, right, proxy_left, proxy_right]
        network = Network(
            left,
            right,
            proxy_left,
            proxy_right,
            None,
            False,
            ["monoResMatch"],
            network_params,
        )

    return network, dataloader, placeholders


def makedirs(path):
    """Create a dir if not exists
    Args:
        path: path to dir that you want to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def train(network):
    print(" [*] Training....")

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    makedirs(args.logd_directory)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    training_flag = tf.placeholder(tf.bool)
    learning_rate_schedule = [int(i) for i in args.learning_rate_schedule.split(",")]
    tf.summary.scalar(
        "learning_rate", learning_rate, collections=network.model_collection
    )

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(network.loss)
    saver = tf.train.Saver(max_to_keep=args.max_to_keep)
    summary_op = tf.summary.merge_all(network.model_collection[0])
    writer = tf.summary.FileWriter(args.log_directory + "/summary/", graph=sess.graph)

    global_step = tf.Variable(0, trainable=False)
    total_num_parameters = 0
    vars = [k for k in tf.trainable_variables()]
    for variable in vars:
        total_num_parameters += np.array(variable.get_shape().as_list()).prod()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coordinator)
    print(" [*] Number of trainable parameters: {}".format(total_num_parameters))

    print(" [*] Training data loaded successfully")
    lr = args.initial_learning_rate

    if args.checkpoint_path != "":
        saver.restore(sess, args.checkpoint_path)
        print(" [*] Load model: SUCCESS")
        if args.retrain:
            sess.run(global_step.assign(0))
        else:
            sess.run(
                global_step.assign(
                    int(os.path.basename(args.checkpoint_path).split("-")[1])
                )
            )

    start_step = global_step.eval(session=sess)

    print(" [*] Start Training...")
    for step in range(start_step, args.iterations):
        before_op_time = time.time()
        _, loss = sess.run(
            [optimizer, network.loss],
            feed_dict={learning_rate: lr, training_flag: True},
        )
        duration = time.time() - before_op_time

        if step and step % args.display_step == 0:
            examples_per_sec = args.batch_size / duration
            training_time_left = (
                ((args.iterations - step) / examples_per_sec) * args.batch_size / 3600.0
            )

            print(
                "Step: [%2d]" % step
                + "/[%2d]" % args.iterations
                + ", Loss: [%2f]" % loss
                + ", Examples/s: [%2f]" % examples_per_sec
                + ", Time left: [%2f]" % training_time_left
            )

            summary_str = sess.run(
                summary_op, feed_dict={learning_rate: lr, training_flag: True}
            )
            writer.add_summary(summary_str, global_step=step)

        if step % args.save_iter_freq == 0:
            saver.save(
                sess, args.log_directory + "/" + args.model_name, global_step=step
            )

        if step in learning_rate_schedule:
            lr = lr / args.learning_rate_scale_factor

    saver.save(
        sess, args.log_directory + "/" + args.model_name, global_step=args.iterations
    )

    print("[*] done")

    coordinator.request_stop()
    coordinator.join(threads)


def test(network, dataloader, placeholders):
    print("\n [*] Testing....")

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    makedirs(args.output_path)

    training_flag = tf.placeholder(tf.bool)
    saver = tf.train.Saver()

    init_op = tf.group(
        tf.global_variables_initializer(), tf.local_variables_initializer()
    )
    sess.run(init_op)

    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # NOTE: in the dataloader, we stack the flipped image with the original image
    # this means that now we have to flip again the first one, and let untouched the
    # second one
    disp_0 = tf.image.flip_left_right(network.disp_left_est_refined[0][0:1, :, :,])
    disp_1 = network.disp_left_est_refined[0][1:2, :, :]

    with open(args.filenames_file, "r") as f:
        names = f.readlines()
    split = 1 if args.right else 0
    names = [n.strip().split(" ")[split] for n in names]

    if args.checkpoint_path != "":
        saver.restore(sess, args.checkpoint_path)
        print(" [*] Load model: SUCCESS")

    num_test_samples = count_text_lines(args.filenames_file)
    dest = args.output_path + args.temp_folder
    with tqdm(total=num_test_samples) as main_bar:
        for step in range(num_test_samples):
            left_im, right_im, left_pr, right_pr, height, width, name = sess.run(
                [
                    dataloader.left_image_batch,
                    dataloader.right_image_batch,
                    dataloader.proxy_left_batch,
                    dataloader.proxy_right_batch,
                    dataloader.image_h,
                    dataloader.image_w,
                    dataloader.name,
                ],
                feed_dict={training_flag: False},
            )
            if args.number_hypothesis == -1:
                # NOTE: do not apply consensus mechanism.
                # We have to use the prediction for the original image,
                # and not the flipped one
                disp = sess.run(
                    disp_1,
                    feed_dict={
                        placeholders[0]: left_im,
                        placeholders[1]: right_im,
                        placeholders[2]: left_pr,
                        placeholders[3]: right_pr,
                        training_flag: False,
                    },
                )
                disp = cv2.resize(
                    disp[0], (width, height), interpolation=cv2.INTER_LINEAR
                ) * (width / args.width)
                name = os.path.basename(name)
                cv2.imwrite(
                    os.path.join(args.output_path, name.decode("utf-8")),
                    (disp * 256.0).astype(np.uint16),
                )
                main_bar.update(1)
                continue

            # NOTE: apply consensus mechanism.
            # In a single run we have obtained both the predictions for the flipped and the original
            # image. We have to save them and apply the consensus mechanism
            with tqdm(total=args.number_hypothesis) as bar:
                for n in range(args.number_hypothesis):
                    disp_flip, disp = sess.run(
                        [disp_0, disp_1],
                        feed_dict={
                            placeholders[0]: left_im,
                            placeholders[1]: right_im,
                            placeholders[2]: left_pr,
                            placeholders[3]: right_pr,
                            training_flag: False,
                        },
                    )

                    makedirs(dest)

                    # Resize and scale images
                    _resize_and_save(
                        disp, width=width, height=height, step=n, dest=dest
                    )
                    _resize_and_save(
                        disp_flip,
                        width=width,
                        height=height,
                        step=n + args.number_hypothesis,
                        dest=dest,
                    )
                    bar.update(1)
            consensus.consensus_mechanism(
                name=names[step],
                multiple_predictions_folder=dest,
                destination=args.output_path,
                number_hypothesis=args.number_hypothesis * 2,
            )
            print("Write disparity [%2d] filtered using consensus mechanism" % step)
            main_bar.update(1)
    print("done.")

    coordinator.request_stop()
    coordinator.join(threads)


def _resize_and_save(disp, width, height, step, dest):
    """Resize the prediction and save it as 16 bit image
    Args:
        disp: prediction to save
        width: original width of the image
        height: original height of the image
        step: current step
        dest: destination folder
    Return:
        save the disp as 16 bit png image into dest folder. The name is the step
    """
    disp = cv2.resize(
        disp.squeeze(), (width, height), interpolation=cv2.INTER_LINEAR,
    ) * (width / args.width)

    cv2.imwrite(
        os.path.join(dest, str(step) + ".png"), (disp * 256.0).astype(np.uint16),
    )


def main(_):

    network_params, dataloader_params = configure_parameters()
    monoResMatch, dataloader, placeholders = configure_network(
        network_params, dataloader_params
    )

    if args.is_training:
        train(monoResMatch)
    else:
        test(monoResMatch, dataloader, placeholders)


if __name__ == "__main__":
    tf.app.run()
