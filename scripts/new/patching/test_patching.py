import argparse
import os
import sys
from math import sqrt

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from scripts.models.lf import lf_dataset
from scripts.models.lol import lol_dataset
from scripts.models.lol.lol_dataset import LolDataset

from scripts.models.lol.lol_model_patching import LineOutlinerTsa
from scripts.new.patching.extract_tensor_patch import extract_tensor_patch
from scripts.utils.dataset_parser import load_file_list_direct
from scripts.utils.files import create_folders
from scripts.utils.wrapper import DatasetWrapper

parser = argparse.ArgumentParser(description='Prepare data for training')
parser.add_argument("--dataset", default="iam")
parser.add_argument("--batch_size", default=1)
parser.add_argument("--images_per_epoch", default=1000)
parser.add_argument("--stop_after_no_improvement", default=10)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--tsa_size", default=3)
parser.add_argument("--patch_ratio", default=5)
parser.add_argument("--output", default="scripts/original/snapshots/training")
args = parser.parse_args()

print("[Training Line-Follower] Model: SFR | Dataset: ", args.dataset)

data_folder = os.getenv("DATA_FOLDER") if os.getenv("DATA_FOLDER") else "data"

target_folder = os.path.join(data_folder, "sfrs", args.dataset)
pages_folder = os.path.join(target_folder, "pages")
char_set_path = os.path.join(pages_folder, "character_set.json")

training_set_list_path = os.path.join(pages_folder, "training.json")
training_set_list = load_file_list_direct(training_set_list_path)
train_dataset = LolDataset(training_set_list, augmentation=True)
train_dataloader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=lol_dataset.collate)
batches_per_epoch = int(args.images_per_epoch / args.batch_size)
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

dtype = torch.cuda.FloatTensor

initial = torch.tensor([10.0, 10.0]).cuda()
current_angle = torch.tensor(45.0).cuda()
rotation = torch.tensor(
    [[torch.cos(torch.deg2rad(current_angle)), 1.0 * torch.sin(torch.deg2rad(current_angle))],
     [-1.0 * torch.sin(torch.deg2rad(current_angle)), torch.cos(torch.deg2rad(current_angle))]]).cuda()
rotated = torch.matmul(initial, rotation.t())

current_scale = 1 / (sqrt(2))
scaling = torch.tensor([[current_scale, 0.], [0., current_scale]]).cuda()

scaled = torch.matmul(rotated, scaling)

assert scaled[0].item() == 10

t1 = torch.tensor([[[10, 10], [20, 20], [30, 30], [10, 0], [0.5, 0]]])
t2 = torch.tensor([[[20, 10], [20, 20], [30, 30], [5, 0], [0.5, 0]]])

loss = torch.nn.MSELoss(reduction="sum")(t1, t2)


print(loss)
print(loss)

for epoch in range(1000):

    for index, x in enumerate(train_dataloader):
        x = x[0]
        img = Variable(x['img'].type(dtype), requires_grad=False)[None, ...]
        ground_truth = x["steps"]

        folder = os.path.join("screenshots")

        np_img = img[0].clone().detach().cpu().numpy().transpose()
        np_img = (np_img + 1) * 128.0

        cv2.imwrite(os.path.join(folder, "full.png"), np_img)

        for step_index, step in enumerate(ground_truth):
            patch_name = os.path.join(folder, str(step_index) + ".png")

            initial_step = step
            current_height = torch.dist(initial_step[1], initial_step[0]).cuda()
            current_scale = 64 / (current_height * 5).cuda()
            current_angle = initial_step[3][0].cuda()

            current_base = torch.stack([initial_step[1][0], initial_step[1][1]]).cuda()
            patch_parameters = torch.stack([current_base[0],  # x
                                            current_base[1],  # y
                                            torch.deg2rad(current_angle),
                                            # torch.remainder(current_angle, torch.tensor(360)),  # angle
                                            current_height]).unsqueeze(0)
            patch_parameters = patch_parameters.cuda()
            patch = extract_tensor_patch(img, patch_parameters, size=64)
            patch = patch[0].clone().detach().cpu().numpy().transpose()

            patch = (patch + 1) * 128.0
            create_folders(patch_name)
            cv2.imwrite(patch_name, patch)

        sys.exit(0)
