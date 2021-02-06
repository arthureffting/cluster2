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
from scripts.models.lol.lol_dataset import LolDataset

from scripts.models.lol.lol_model_patching import LineOutlinerTsa
from scripts.models.lol.loss import standard
from scripts.new.training.run_model import paint_model_run
from scripts.utils.dataset_parser import load_file_list_direct
from scripts.utils.files import create_folders, save_to_json
from scripts.utils.wrapper import DatasetWrapper

parser = argparse.ArgumentParser(description='Prepare data for training')
parser.add_argument("--dataset", default="iam")
parser.add_argument("--batch_size", default=1)
parser.add_argument("--images_per_epoch", default=5000)
parser.add_argument("--stop_after_no_improvement", default=20)
parser.add_argument("--learning_rate", default=0.0001)

# Patching
parser.add_argument("--tsa_size", default=3)
parser.add_argument("--patch_ratio", default=5)
parser.add_argument("--patch_size", default=64)

# Training techniques
parser.add_argument("--name", default="other")
parser.add_argument("--mode", default="reset_threshold")
parser.add_argument("--save-every", default=200)
parser.add_argument("--reset-threshold", default=25)
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
batches_per_epoch = int(args.images_per_epoch / args.batch_size)
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list_path = os.path.join(pages_folder, "testing.json")
test_set_list = load_file_list_direct(test_set_list_path)
test_dataset = LolDataset(test_set_list)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=lol_dataset.collate)
test_dataloader = DatasetWrapper(test_dataloader, 400)

validation_path = os.path.join(pages_folder, "validation.json")
validation_list = load_file_list_direct(validation_path)
validation_set = LolDataset(validation_list[0:1])
validation_loader = DataLoader(validation_set,
                               batch_size=1,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=lol_dataset.collate)

print("Loaded datasets")

lol = LineOutlinerTsa(tsa_size=args.tsa_size, patch_size=args.patch_size, patch_ratio=args.patch_ratio)
lol.cuda()

optimizer = torch.optim.Adam(lol.parameters(), lr=args.learning_rate)

dtype = torch.cuda.FloatTensor

lowest_loss = np.inf
cnt_since_last_improvement = 0

for epoch in range(1000):

    epoch_data = {
        "epoch": epoch,
    }

    print("Epoch", epoch)

    sum_loss = 0.0
    steps = 0.0

    lol.train()

    max_steps_ran = 0
    ran_steps_sum = 0

    for index, x in enumerate(train_dataloader):
        # Only single batch for now
        x = x[0]
        img = Variable(x['img'].type(dtype), requires_grad=False)[None, ...]
        ground_truth = x["steps"]

        # Picks a random place in the line to start
        sol_index = 0 if not args.random_sol else random.choice(range(len(ground_truth) - 4))
        sol = ground_truth[sol_index]
        ground_truth_used = ground_truth[sol_index:]

        predicted_steps, length, _ = lol(img,
                                         sol,
                                         ground_truth_used,
                                         reset_threshold=25 if args.mode == "reset_threshold" else None,
                                         max_steps=3 if args.mode == "max_steps" else None,
                                         disturb_sol=True)

        ran_steps_sum += length
        max_steps_ran = length if length > max_steps_ran else max_steps_ran

        if predicted_steps is None:
            continue

        ground_truth_for_steps = ground_truth_used[1:1 + length].cuda()

        loss = standard(predicted_steps, ground_truth_for_steps)
        # loss = torch.nn.MSELoss(reduction="sum")(predicted_steps, ground_truth_for_steps.cuda())
        # loss =

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        steps += 1

        sys.stdout.write("\r[Training] " + str(1 + index) + "/" + str(len(train_dataloader)) + " | " + str(
            round(sum_loss / steps, 2)) + " | count: " + str(length) + " | max: " + str(
            max_steps_ran) + " | avg: " + str(
            round(ran_steps_sum / steps, 1)))

        if index % args.save_every == 0:
            model_path = os.path.join(args.output, args.name, 'last.pt')
            screenshot_path = os.path.join(args.output, args.name, "screenshots",
                                           str(epoch) + "-" + str(index) + ".png")
            create_folders(screenshot_path)
            torch.save(lol.state_dict(), model_path)
            time.sleep(1)
            paint_model_run(model_path, validation_loader, destination=screenshot_path)

    epoch_data["train"] = {
        "loss": sum_loss / steps,
        "max_steps": max_steps_ran,
        "avg_steps": ran_steps_sum / steps
    }

    print("\nTrain Loss", sum_loss / steps)
    print("Real Epoch", train_dataloader.epoch)

    sum_loss = 0.0
    steps = 0.0
    max_steps_ran = 0
    ran_steps_sum = 0

    lol.eval()

    for index, x in enumerate(test_dataloader):
        x = x[0]
        img = Variable(x['img'].type(dtype), requires_grad=False)[None, ...]
        ground_truth = x["steps"]

        # If using max steps, run the same as training
        # Picks a random place in the line to start
        sol_index = 0 if not args.random_sol else random.choice(range(len(ground_truth) - 4))
        sol = ground_truth[sol_index]
        ground_truth_used = ground_truth[sol_index:]
        predicted_steps, length, _ = lol(img,
                                         sol,
                                         ground_truth_used,
                                         reset_threshold=25 if args.mode == "reset_threshold" else None,
                                         max_steps=3 if args.mode == "max_steps" else None,
                                         disturb_sol=False)
        ran_steps_sum += length
        max_steps_ran = length if length > max_steps_ran else max_steps_ran
        if predicted_steps is None:
            continue
        ground_truth_for_steps = ground_truth_used[1:1 + length].cuda()
        loss = standard(predicted_steps, ground_truth_for_steps)

        loss = loss / length
        sum_loss += loss.item()
        steps += 1
        sys.stdout.write("\r[Testing] " + str(1 + index) + "/" + str(len(test_dataloader)) + " | " + str(
            round(sum_loss / steps, 2)) + " | count: " + str(length) + " | max: " + str(
            max_steps_ran) + " | avg: " + str(
            round(ran_steps_sum / steps, 1)))

    cnt_since_last_improvement += 1

    epoch_data["test"] = {
        "loss": sum_loss / steps,
        "max_steps": max_steps_ran,
        "avg_steps": ran_steps_sum / steps
    }

    if lowest_loss > sum_loss / steps:
        cnt_since_last_improvement = 0
        lowest_loss = sum_loss / steps
        print("\nSaving Best")
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        torch.save(lol.state_dict(), os.path.join(args.output, args.name, 'best.pt'))

    epoch_json_path = os.path.join(args.output, args.name, "epochs", str(epoch) + ".json")
    create_folders(epoch_json_path)
    save_to_json(epoch_data, epoch_json_path)

    print("\nTest Loss", sum_loss / steps, lowest_loss)
    print()

    if cnt_since_last_improvement >= args.stop_after_no_improvement:
        break