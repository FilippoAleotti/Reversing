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
from torch.autograd import Variable
from tqdm import tqdm
from utils import general
from dataset import factory as dataset_factory
from dataloader import factory as dataloader_factory


def run_test(model, configuration):
    """Generate image for test the network
    """
    configuration.parse_test_params()
    general_params = configuration.general_params
    test_params = configuration.test_params
    padding_params = configuration.padding_params

    print("=> Dataset: " + test_params["dataset"])
    print(
        "=> final height: {}  final width: {}".format(
            padding_params["final_h"], padding_params["final_w"]
        )
    )
    filepath = dataset_factory.get_test_file_path(
        test_params["dataset"], test_params["datapath"]
    )
    dataset = dataset_factory.get_dataset_test(test_params["dataset"])
    all_left, all_right = dataset(filepath)
    loader = dataloader_factory.get_dataloader(test_params["dataset"])(
        all_left,
        all_right,
        mode="test",
        params={**test_params, **padding_params},
        augmentation_params=None,
    )
    test_dataloader = torch.utils.data.DataLoader(
        loader, batch_size=1, shuffle=False, num_workers=4, drop_last=False
    )

    # load checkpoint
    if not general_params["cuda"]:
        print("=> loading model on CPU")
        state_dict = torch.load(test_params["ckpt"], map_location=torch.device("cpu"))
    else:
        state_dict = torch.load(test_params["ckpt"])
    model.load_state_dict(state_dict["state_dict"], strict=True)

    model.eval()
    num_test_samples = len(test_dataloader)
    with tqdm(total=num_test_samples) as bar:
        for _, (left, right, original_shape, name) in enumerate(test_dataloader):
            if general_params["cuda"]:
                left = torch.FloatTensor(left).cuda()
                right = torch.FloatTensor(right).cuda()
            left, right = Variable(left), Variable(right)
            with torch.no_grad():
                output = model(left, right)
                output = torch.squeeze(output)
                output = torch.nn.functional.relu(output)
                output = output.data.cpu().numpy()
                original_h = original_shape[0].data.cpu().numpy()[0]
                original_w = original_shape[1].data.cpu().numpy()[0]
                name = name[0]
                extension = "." + name.split(".")[-1]
                name = name.replace(extension, "")

            top_pad = padding_params["final_h"] - original_h
            left_pad = padding_params["final_w"] - original_w
            final_output = output[top_pad:, :-left_pad]
            if (
                final_output.shape[0] != original_shape[0]
                or final_output.shape[1] != original_shape[1]
            ):
                raise ValueError("Problems with shape")

            destination = os.path.join(
                test_params["results"],
                "{}",
                test_params["model"],
                test_params["dataset"],
                name + ".png",
            )

            general.save_kitti_disp(destination.format("16bit"), final_output)
            if test_params["qualitative"]:
                general.save_color(
                    destination.format("qualitative"),
                    final_output,
                    cmap=test_params["cmap"],
                )

            bar.update(1)
