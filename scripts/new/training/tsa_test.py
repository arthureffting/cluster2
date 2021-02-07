import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.models.lol import lol_dataset
from scripts.models.lol.lol_dataset import LolDataset
from scripts.models.lol.lol_model_patching_alt import LineOutlinerTsa
from scripts.models.lol.loss import distributed_weights
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

lol = LineOutlinerTsa(tsa_size=5, path=os.path.join("scripts", "new", "snapshots", "lol-sep", "separated-loss", "last.pt"))
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

    sol = ground_truth[0]

    predicted_steps, length, input = lol(img, sol, ground_truth, max_steps=30, disturb_sol=False)


    vertical_concats = []
    for tsa_line in input:
        for tsa_image in tsa_line:
            horizontal_concats = []
            for tsa_section in tsa_image:
                img_np = tsa_section.clone().detach().cpu().numpy().transpose()
                img_np = (img_np + 1) * 128
                horizontal_concats.append(img_np)
                horizontal_concats.append(np.zeros((64, 2, 3), dtype=np.float32))
            vertical_concats.append(cv2.hconcat(horizontal_concats))
            vertical_concats.append(np.zeros((2, (args.tsa_size * 2) + (64 * args.tsa_size), 3), dtype=np.float32))
    s_path = os.path.join("screenshots", "tsa", str(counter) + ".png")
    cv2.imwrite(s_path, cv2.vconcat(vertical_concats))
    create_folders(s_path)
    counter += 1

    # GROUND TRUTH
    # ground_truth_upper_steps = [Point(step[0][0].item(), step[0][1].item()) for step in ground_truth]
    # ground_truth_baseline_steps = [Point(step[1][0].item(), step[1][1].item()) for step in ground_truth]
    # ground_truth_lower_steps = [Point(step[2][0].item(), step[2][1].item()) for step in ground_truth]
    # for i in range(len(ground_truth_upper_steps)):
    #     painter.draw_line(
    #         [ground_truth_upper_steps[i], ground_truth_baseline_steps[i], ground_truth_lower_steps[i]],
    #         color=(0, 0, 0, 1), line_width=2)
    #
    # painter.draw_line(ground_truth_upper_steps, line_width=4, color=(0, 0, 0, 0.5))
    # painter.draw_line(ground_truth_baseline_steps, line_width=4, color=(0, 0, 0, 0.5))
    # painter.draw_line(ground_truth_lower_steps, line_width=4, color=(0, 0, 0, 0.5))

    upper_steps = [Point(step[0][0].item(), step[0][1].item()) for step in predicted_steps]
    baseline_steps = [Point(step[1][0].item(), step[1][1].item()) for step in predicted_steps]
    lower_steps = [Point(step[2][0].item(), step[2][1].item()) for step in predicted_steps]
    confidences = [step[4][0].item() for step in predicted_steps]
    angles = [step[3][0].item() for step in predicted_steps]

    for i in range(len(baseline_steps)):
        painter.draw_line([upper_steps[i], baseline_steps[i], lower_steps[i]], color=(0, 0, 1, 1), line_width=2)

    for index, step in enumerate(baseline_steps[:-1]):
        upper = upper_steps[index]
        lower = lower_steps[index]
        next_step = baseline_steps[index + 1]
        next_upper = upper_steps[index + 1]
        next_lower = lower_steps[index + 1]
        confidence = confidences[index]
        painter.draw_area([upper, next_upper, next_step, next_lower, lower, step], line_color=(0, 0, 0, 0),
                          line_width=0,
                          fill_color=(1, 0, 0, confidence))

    painter.draw_line(baseline_steps, line_width=4, color=(0, 0, 1, 1))
    painter.draw_line(upper_steps, line_width=4, color=(1, 0, 1, 1))
    painter.draw_line(lower_steps, line_width=4, color=(1, 0, 1, 1))

    for index, step in enumerate(baseline_steps):
        painter.draw_point(step, radius=6)
        predicted_angle_projection = get_new_point(step, angles[index], 30)
        painter.draw_line([step, predicted_angle_projection], color=(0, 1, 0, 1), line_width=3)

    sol = {
        "upper_point": ground_truth[0][0],
        "base_point": ground_truth[0][1],
        "angle": ground_truth[0][3][0],
    }

    sol_upper = Point(sol["upper_point"][0].item(), sol["upper_point"][1].item())
    sol_lower = Point(sol["base_point"][0].item(), sol["base_point"][1].item())

    painter.draw_line([sol_lower, sol_upper], color=(0, 1, 0, 1), line_width=3)
    painter.draw_point(sol_lower, color=(0, 1, 0, 1), radius=3)
    painter.draw_point(sol_upper, color=(0, 1, 0, 1), radius=3)

destination = os.path.join("screenshots", "tsa", "full.png")
create_folders(destination)
painter.save(destination)
