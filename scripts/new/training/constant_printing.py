import argparse
import os
import time

from torch.utils.data import DataLoader

from scripts.models.lol import lol_dataset
from scripts.models.lol.lol_dataset import LolDataset
from scripts.new.training.run_model import paint_model_run
from scripts.utils.dataset_parser import load_file_list_direct

parser = argparse.ArgumentParser(description='Prepare data for training')
parser.add_argument("--dataset", default="iam")
parser.add_argument("--batch_size", default=1)
parser.add_argument("--images_per_epoch", default=5000)
parser.add_argument("--stop_after_no_improvement", default=20)
parser.add_argument("--learning_rate", default=0.0002)
parser.add_argument("--tsa_size", default=3)
parser.add_argument("--patch_ratio", default=5)
parser.add_argument("--output", default="scripts/original/snapshots/training")
parser.add_argument("--model", default="scripts/new/snapshots/training2/lol-last.pt")
args = parser.parse_args()

data_folder = os.getenv("DATA_FOLDER") if os.getenv("DATA_FOLDER") else "data"
target_folder = os.path.join(data_folder, "sfrs", args.dataset)
pages_folder = os.path.join(target_folder, "pages")
char_set_path = os.path.join(pages_folder, "character_set.json")

test_set_list_path = os.path.join(pages_folder, "validation.json")
test_set_list = load_file_list_direct(test_set_list_path)
test_dataset = LolDataset(test_set_list[0:1])
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=lol_dataset.collate)

count = 0

while True:
    for t in ["training", "training2"]:
        model_path = "scripts/new/snapshots/" + t + "/last.pt"
        paint_model_run(model_path, test_dataloader, destination=os.path.join("screenshots", t, str(count) + ".png"))
    time.sleep(30)
    count += 1
