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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils import image_plotter
from loss.supervised import Loss
from utils import general
from dataset import factory as dataset_factory
from dataloader import factory as dataloader_factory


def run_train(model, configuration):
    """ train the network """
    configuration.parse_train_params()
    general_params = configuration.general_params
    dataset_params = configuration.dataset_params
    train_params = configuration.train_params
    log_params = configuration.log_params
    loss_params = configuration.loss_params
    augmentation_params = configuration.augmentation_params

    logdir = os.path.join(train_params["savemodel"], general_params["model"])
    train_writer = SummaryWriter(logdir)

    image_writer = image_plotter.Plotter(train_writer)
    n_iter = 0
    starting_epoch = 0
    general.run_tensorboard(logdir=logdir, port=log_params["training_port"])

    optimizer = optim.Adam(
        model.parameters(), lr=train_params["initial_learning_rate"], betas=(0.9, 0.999)
    )

    # selecting dataset
    print("=> Dataset: " + str(dataset_params["dataset"]))
    filename = dataset_params["filename"]
    training_dataset = dataset_factory.get_dataset_train(dataset_params["dataset"])
    all_left_img, all_right_img, all_proxy_left, all_proxy_right = training_dataset(
        filename, rgb_ext=dataset_params["rgb_ext"], proxy_ext=".png",
    )

    # training dataloader
    dataloader = dataloader_factory.get_dataloader(dataset_params["dataset"])
    training_dataloader = torch.utils.data.DataLoader(
        dataloader(
            all_left_img,
            all_right_img,
            mode="training",
            params=dataset_params,
            proxy_left=all_proxy_left,
            proxy_right=all_proxy_right,
            augmentation_params=augmentation_params,
        ),
        batch_size=train_params["batch"],
        shuffle=True,
        num_workers=8,
        drop_last=False,
    )

    epoch_size = len(training_dataloader)

    if train_params["ckpt"] is not None:
        state_dict = torch.load(train_params["ckpt"])
        model.load_state_dict(state_dict["state_dict"], strict=True)
        loaded_epoch = state_dict["epoch"]
        if "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])
        print("=> restored model trained for {} epochs".format(loaded_epoch))

        loaded_epoch += 1
        print("=> train must restart from epoch {}".format(loaded_epoch))
        starting_epoch = loaded_epoch
        n_iter = (loaded_epoch * epoch_size) + 1

    # NOTE: save a copy of the training configuration
    training_cfg = {
        "n_iter": n_iter,
        "starting_epoch": starting_epoch,
    }
    training_cfg = {
        **training_cfg,
        **train_params,
        **dataset_params,
        **loss_params,
        **augmentation_params,
    }
    general.write_cfg(
        params=training_cfg,
        dest=os.path.join(
            train_params["savemodel"], train_params["model"], "train_cfg"
        ),
    )

    train_writer.add_scalar("batch_size", train_params["batch"], 0)
    for i, w in enumerate(loss_params["weights"]):
        train_writer.add_scalar("weight_scale_" + str(i), w, 0)

    loss_function = Loss(loss_weights=loss_params["weights"])
    for epoch in range(starting_epoch, train_params["epochs"]):
        print("This is %d-th epoch" % (epoch))

        # adjusting learning rate
        lr = train_params["initial_learning_rate"]
        if epoch >= train_params["milestone"]:
            lr = train_params["initial_learning_rate"] * train_params["decay_factor"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        ## training ##
        model.train()
        for batch_idx, (left, right, proxy) in enumerate(training_dataloader):
            left = Variable(torch.FloatTensor(left))
            right = Variable(torch.FloatTensor(right))
            proxy = Variable(torch.FloatTensor(proxy))
            left, right, proxy = left.cuda(), right.cuda(), proxy.cuda()
            optimizer.zero_grad()
            predictions = model(left, right)
            loss = loss_function(predictions=predictions, target=proxy)
            loss.backward()
            optimizer.step()

            if batch_idx % log_params["print_freq"] == 0 and batch_idx > 0:

                train_writer.add_scalar("train_loss_total_training", loss, n_iter)
                train_writer.add_scalar(
                    "epoch_{}_train_loss".format(epoch), loss, batch_idx
                )

                for param_group in optimizer.param_groups:
                    train_writer.add_scalar("learning_rate", param_group["lr"], n_iter)

                summary_left = image_writer.prepare_summary(
                    "images/left/epoch_{}".format(epoch),
                    left,
                    max_el=log_params["max_el"],
                )
                summary_right = image_writer.prepare_summary(
                    "images/right/epoch_{}".format(epoch),
                    right,
                    max_el=log_params["max_el"],
                )
                summary_proxy = image_writer.prepare_summary(
                    "proxy/epoch_{}".format(epoch),
                    proxy,
                    disp=True,
                    max_el=log_params["max_el"],
                )
                image_writer.plot_summary_images(summary_left, batch_idx)
                image_writer.plot_summary_images(summary_right, batch_idx)
                image_writer.plot_summary_images(summary_proxy, batch_idx)

                for i, pred in enumerate(predictions):
                    summary_prediction = image_writer.prepare_summary(
                        "epoch_{}/scale_{}".format(epoch, i),
                        pred,
                        disp=True,
                        max_el=log_params["max_el"],
                    )
                    image_writer.plot_summary_images(summary_prediction, batch_idx)
                print(
                    "Iter {}/{} | loss = {:.3f} |".format(batch_idx, epoch_size, loss)
                )

            n_iter += 1
            if batch_idx >= epoch_size:
                break

        print("epoch %d ended!" % (epoch))

        # save checkpoint
        savefilename = os.path.join(
            train_params["savemodel"],
            train_params["model"],
            "final_epoch_" + str(epoch) + ".tar",
        )
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            savefilename,
        )

    print("Done! Training is ended")
