import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from scripts.models.lol import lol_dataset
from scripts.models.lol.evaluation import evaluation_cost
from scripts.models.lol.lol_dataset import LolDataset

from scripts.models.lol.lol_model_patching_alt import LineOutlinerTsa
from scripts.models.lol.loss import outline_loss, stop_loss, distributed_weights, separate_losses
from scripts.new.training.run_model import paint_model_run
from scripts.utils.dataset_parser import load_file_list_direct
from scripts.utils.files import create_folders, save_to_json
from scripts.utils.wrapper import DatasetWrapper

parser = argparse.ArgumentParser(description='Prepare data for training')
parser.add_argument("--dataset", default="iam")
parser.add_argument("--batch_size", default=1)
parser.add_argument("--images_per_epoch", default=10)
parser.add_argument("--testing_images_per_epoch", default=10)
parser.add_argument("--stop_after_no_improvement", default=50)
parser.add_argument("--learning_rate", default=0.001)

# Patching
parser.add_argument("--tsa_size", default=5)
parser.add_argument("--patch_ratio", default=5)
parser.add_argument("--patch_size", default=64)
parser.add_argument("--min_height", default=32)

# Training techniques
parser.add_argument("--name", default="weighted-loss")
parser.add_argument("--reset-threshold", default=25)
parser.add_argument("--max_steps", default=6)
parser.add_argument("--random-sol", default=True)

parser.add_argument("--output", default="scripts/new/snapshots/lol")
args = parser.parse_args()

args_filename = os.path.join(args.output, args.name, 'args.json')
create_folders(args_filename)
with open(args_filename, 'w') as fp:
    json.dump(args.__dict__, fp, indent=4)

print("[Training Line-Outliner] Model: SFRS | Dataset: ", args.dataset)

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
batches_per_epoch = int(int(args.images_per_epoch) / args.batch_size)
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list_path = os.path.join(pages_folder, "testing.json")
test_set_list = load_file_list_direct(test_set_list_path)
test_dataset = LolDataset(test_set_list)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=lol_dataset.collate)
test_dataloader = test_dataloader if args.testing_images_per_epoch is None else DatasetWrapper(test_dataloader,
                                                                                               int(
                                                                                                   args.testing_images_per_epoch))

validation_path = os.path.join(pages_folder, "validation.json")
validation_list = load_file_list_direct(validation_path)
validation_set = LolDataset(validation_list[0:1])
validation_loader = DataLoader(validation_set,
                               batch_size=1,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=lol_dataset.collate)

print("Loaded datasets")

lol = LineOutlinerTsa(tsa_size=args.tsa_size,
                      patch_size=args.patch_size,
                      min_height=args.min_height,
                      patch_ratio=args.patch_ratio)
lol.cuda()

optimizer = torch.optim.Adam(lol.parameters(), lr=float(args.learning_rate))

dtype = torch.cuda.FloatTensor

best_loss = 0.0
cnt_since_last_improvement = 0

for epoch in range(1000):

    epoch_data = {
        "epoch": epoch,
    }

    print("[Epoch", epoch, "]")

    sum_loss = 0.0
    steps = 0.0

    lol.train()

    for index, x in enumerate(train_dataloader):
        # Only single batch for now
        x = x[0]
        img = Variable(x['img'].type(dtype), requires_grad=False)[None, ...]
        ground_truth = x["steps"]

        # Iterates over the line until the end
        ran_until = 0

        while ran_until < len(ground_truth) - 2:
            sol = ground_truth[ran_until].cuda()
            ground_truth_used = ground_truth[ran_until:]
            predicted_steps, length, _ = lol(img,
                                             sol,
                                             ground_truth_used,
                                             reset_threshold=25,
                                             disturb_sol=True)

            if length == 0: break
            desired_steps = ground_truth_used[1:1 + length].cuda()
            total_loss = separate_losses(predicted_steps, desired_steps)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            sum_loss += total_loss.item()
            steps += length
            ran_until += length
            sys.stdout.write("\r[Training] " + str(1 + index) + "/" + str(len(train_dataloader)) + " | mse: " + str(
                int(sum_loss / steps)))

    print()

    epoch_data["train"] = {
        "loss": sum_loss / steps
    }

    sum_loss = 0.0
    steps = 0.0

    lol.eval()

    # Save epoch snapshot using some validation image
    model_path = os.path.join(args.output, args.name, 'last.pt')
    screenshot_path = os.path.join(args.output, args.name, "screenshots", str(epoch) + ".png")
    create_folders(screenshot_path)
    torch.save(lol.state_dict(), model_path)
    time.sleep(1)
    paint_model_run(model_path, validation_loader, destination=screenshot_path)

    for index, x in enumerate(test_dataloader):
        x = x[0]
        img = Variable(x['img'].type(dtype), requires_grad=False)[None, ...]
        ground_truth = x["steps"]

        # Iterates over the line until the end

        sol = ground_truth[0].cuda()
        predicted_steps, length, _ = lol(img,
                                         sol,
                                         ground_truth,
                                         max_steps=len(ground_truth) - 1,
                                         disturb_sol=False)

        if length == 0: break
        desired_steps = ground_truth[1:1 + length].cuda()
        sum_loss += evaluation_cost(torch.cat([sol.unsqueeze(0), predicted_steps]),
                                    torch.cat([sol.unsqueeze(0), desired_steps]))
        steps += 1
        sys.stdout.write(
            "\r[Testing] " + str(1 + index) + "/" + str(len(test_dataloader)) + " | dice: " + str(
                round(sum_loss / steps, 3)))

    cnt_since_last_improvement += 1

    epoch_data["test"] = {
        "loss": sum_loss / steps
    }

    # How many steps could it run without going too far from the ground truth

    loss_used = (sum_loss / steps)

    if loss_used > best_loss:
        cnt_since_last_improvement = 0
        best_loss = loss_used
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        torch.save(lol.state_dict(), os.path.join(args.output, args.name, 'best.pt'))
        print("\n[New best achieved]")
    else:
        print("\n[Current best]: ", round(best_loss, 3))

    epoch_json_path = os.path.join(args.output, args.name, "epochs", str(epoch) + ".json")
    create_folders(epoch_json_path)
    save_to_json(epoch_data, epoch_json_path)

    print()

    if cnt_since_last_improvement >= args.stop_after_no_improvement:
        break
