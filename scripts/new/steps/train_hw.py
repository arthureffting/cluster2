import argparse
import torch
from torch.autograd import Variable
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
import numpy as np
import json
import os

from scripts.models.hw import hw_dataset, cnn_lstm
from scripts.models.hw.hw_dataset import HwDataset
from scripts.utils import error_rates, string_utils
from scripts.utils.dataset_parser import load_file_list, load_file_list_direct
from scripts.utils.wrapper import DatasetWrapper

parser = argparse.ArgumentParser(description='Prepare data for training')
parser.add_argument("--dataset", default="iam")
parser.add_argument("--input_height", default=32)
parser.add_argument("--learning_rate", default=0.0002)
parser.add_argument("--batch_size", default=4)
parser.add_argument("--images_per_epoch", default=1000)
parser.add_argument("--stop_after_no_improvement", default=10)
parser.add_argument("--output", default="scripts/new/snapshots/hw/default")
args = parser.parse_args()

print("[Training HTR] Model: SFRS | Dataset: ", args.dataset)
data_folder = os.getenv("DATA_FOLDER") if os.getenv("DATA_FOLDER") else "data"

target_folder = os.path.join(data_folder, "sfrs", args.dataset)
pages_folder = os.path.join(target_folder, "pages")
char_set_path = os.path.join(pages_folder, "character_set.json")

with open(char_set_path) as f:
    char_set = json.load(f)

idx_to_char = {}
for k, v in char_set['idx_to_char'].items():
    idx_to_char[int(k)] = v

training_set_list_path = os.path.join(pages_folder, "training.json")
training_set_list = load_file_list_direct(training_set_list_path)
train_dataset = HwDataset(training_set_list,
                          char_set['char_to_idx'], augmentation=True,
                          img_height=args.input_height)
train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True,
                              collate_fn=hw_dataset.collate)
batches_per_epoch = int(args.images_per_epoch / args.batch_size)
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)
testing_set_list_path = os.path.join(pages_folder, "testing.json")
testing_set_list = load_file_list_direct(testing_set_list_path)
test_dataset = HwDataset(testing_set_list,
                         char_set['char_to_idx'],
                         img_height=args.input_height)
test_dataloader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             collate_fn=hw_dataset.collate)

criterion = CTCLoss(blank=0, zero_infinity=True)

hw = cnn_lstm.create_model({
    "num_of_outputs": len(idx_to_char),
    "num_of_channels": 3,
    "cnn_out_size": 512,
    "input_height": args.input_height,
    "char_set_path": char_set_path
}).cpu()

optimizer = torch.optim.Adam(hw.parameters(), lr=args.learning_rate)
lowest_loss = np.inf
cnt_since_last_improvement = 0

dtype = torch.cuda.FloatTensor

for epoch in range(1000):
    print("Epoch", epoch)
    sum_loss = 0.0
    steps = 0.0
    hw.train()
    for i, x in enumerate(train_dataloader):

        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
        labels = Variable(x['labels'], requires_grad=False)
        label_lengths = Variable(x['label_lengths'], requires_grad=False)

        preds = hw(line_imgs.cpu())
        output_batch = preds.permute(1, 0, 2)
        out = output_batch.data.cpu().numpy()

        for i, gt_line in enumerate(x['gt']):
            logits = out[i, ...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)
            sum_loss += cer
            steps += 1

        batch_size = preds.size(1)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        loss = criterion(preds.contiguous(), labels.contiguous(), preds_size.contiguous(), label_lengths.contiguous())
        # print "after"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Train Loss", sum_loss / steps)
    print("Real Epoch", train_dataloader.epoch)

    sum_loss = 0.0
    steps = 0.0
    hw.eval()

    with torch.no_grad():
        for x in test_dataloader:
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
            labels = Variable(x['labels'], requires_grad=False)
            label_lengths = Variable(x['label_lengths'], requires_grad=False)

            preds = hw(line_imgs.cpu()).cpu()
            output_batch = preds.permute(1, 0, 2)
            out = output_batch.data.cpu().numpy()

            for i, gt_line in enumerate(x['gt']):
                logits = out[i, ...]
                pred, raw_pred = string_utils.naive_decode(logits)
                pred_str = string_utils.label2str_single(pred, idx_to_char, False)
                cer = error_rates.cer(gt_line, pred_str)
                sum_loss += cer
                steps += 1

    cnt_since_last_improvement += 1
    if lowest_loss > sum_loss / steps:
        cnt_since_last_improvement = 0
        lowest_loss = sum_loss / steps
        print("Saving Best")

        if not os.path.exists(args.output):
            os.makedirs(args.output)
        torch.save(hw.state_dict(), os.path.join(args.output, 'hw.pt'))

    print("Test Loss", sum_loss / steps, lowest_loss)
    print("")

    if cnt_since_last_improvement >= args.stop_after_no_improvement and lowest_loss < 0.9:
        break
