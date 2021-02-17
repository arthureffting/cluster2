import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.models.lol import lol_dataset
from scripts.models.lol.dice_coefficient.dice_coefficient import DiceCoefficientLoss
from scripts.models.lol.dice_coefficient.utils import upper_polygons, \
    lower_polygons, complete_polygons
from scripts.models.lol.lol_dataset import LolDataset
from scripts.models.lol.lol_lf import LineOutlinerTsa

from scripts.original.iam_conversion.iam_data_loader import Point
from scripts.utils.dataset_parser import load_file_list_direct
from scripts.utils.files import create_folders
from scripts.utils.geometry import get_new_point
from scripts.utils.painter import Painter

dtype = torch.cuda.FloatTensor

img_path = None
painter = None

parser = argparse.ArgumentParser(description='Prepare data for training')
parser.add_argument("--dataset", default="iam")
parser.add_argument("--batch_size", default=1)
parser.add_argument("--images_per_epoch", default=5000)
parser.add_argument("--stop_after_no_improvement", default=20)
parser.add_argument("--learning_rate", default=0.0002)
parser.add_argument("--tsa_size", default=5)
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

# lol = LineOutlinerTsa(tsa_size=5,
#                      path=os.path.join("scripts", "new", "snapshots", "lol", "test-roots", "last.pt"))
lol = LineOutlinerTsa(tsa_size=args.tsa_size)
lol.eval()
lol.cuda()

counter = 0

for index, x in enumerate(test_dataloader):
    x = x[0]

    if img_path is None:
        painter = Painter(path=x["img_path"])
        img_path = x["img_path"]

    belongs = img_path == x["img_path"]

    if not belongs:
        continue

    img = x['img'].type(dtype)[None, ...]
    ground_truth = x["steps"]

    sol = ground_truth[0].cuda()

    predicted_steps, length, input = lol(img, sol, ground_truth, max_steps=len(ground_truth) - 1, disturb_sol=False)
    desired_steps = ground_truth[1:1 + length].cuda()

    # upper_polygon, desired_upper_polygon = complete_polygons(sol, predicted_steps, desired_steps)
    # dice_loss, intersections = DiceCoefficientLoss(upper_polygon, desired_upper_polygon)

    upper_line = [Point(i[2][0].item(), i[2][1].item()) for i in predicted_steps]
    lower_line = [Point(i[0][0].item(), i[0][1].item()) for i in predicted_steps]
    baseline = [Point(i[1][0].item(), i[1][1].item()) for i in predicted_steps]

    painter.draw_line(baseline, color=(0, 0, 1, 1), line_width=3)
    painter.draw_line(upper_line, color=(1, 0, 1, 1), line_width=3)
    painter.draw_line(lower_line, color=(1, 0, 1, 1), line_width=3)

    # for intersection in lower_intersections:
    #     painter.draw_area([Point(p[0].item(), p[1].item()) for p in intersection], fill_color=(0, 1, 0, 0.5),
    #                       line_width=1, line_color=(1, 1, 0, 1))
    #

    #
    # ### DRAWS TSA
    # vertical_concats = []
    # for tsa_line in input:
    #     for tsa_image in tsa_line:
    #         horizontal_concats = []
    #         for tsa_section in tsa_image:
    #             img_np = tsa_section.clone().detach().cpu().numpy().transpose()
    #             img_np = (img_np + 1) * 128
    #             horizontal_concats.append(img_np)
    #             horizontal_concats.append(np.zeros((64, 2, 3), dtype=np.float32))
    #         vertical_concats.append(cv2.hconcat(horizontal_concats))
    #         vertical_concats.append(np.zeros((2, (args.tsa_size * 2) + (64 * args.tsa_size), 3), dtype=np.float32))
    # s_path = os.path.join("screenshots", "tsa", str(counter) + ".png")
    # cv2.imwrite(s_path, cv2.vconcat(vertical_concats))
    # create_folders(s_path)
    counter += 1
    break

destination = os.path.join("screenshots", "tsa", "full.png")
create_folders(destination)
painter.save(destination)
