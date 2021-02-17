import json

import torch
from shapely.geometry import Point
from torch.utils.data import Dataset

from collections import defaultdict
import os
import cv2
import numpy as np
import math
import random

from scripts.new.domain.running_step import RunningStep
from scripts.new.domain.training_step import TrainingStep, angle_between_points
from scripts.utils import safe_load, augmentation


def collate(batch):
    return batch


def get_subdivide_pt(i, pred_full, lf):
    percent = (float(i) + 0.5) / float(len(pred_full))
    lf_percent = (len(lf) - 1) * percent

    lf_idx = int(np.floor(lf_percent))
    step_percent = lf_percent - lf_idx

    x0 = lf[lf_idx]['cx']
    y0 = lf[lf_idx]['cy']
    x1 = lf[lf_idx + 1]['cx']
    y1 = lf[lf_idx + 1]['cy']

    x = x0 * step_percent + x1 * (1.0 - step_percent)
    y = y0 * step_percent + y1 * (1.0 - step_percent)

    return x, y


def step_to_points(step_data):
    return {
        "base_point": Point(step_data["base_point"][0], step_data["base_point"][1]),
        "upper_point": Point(step_data["upper_point"][0], step_data["upper_point"][1]),
        "lower_point": Point(step_data["lower_point"][0], step_data["lower_point"][1]),
        "stop_confidence": step_data["stop_confidence"]
    }


class LolDataset(Dataset):
    def __init__(self, set_list, random_subset_size=None, augmentation=False):
        self.augmentation = augmentation

        self.ids = set_list
        self.ids.sort()

        self.detailed_ids = []
        for ids_idx, paths in enumerate(self.ids):
            json_path, img_path = paths

            d = safe_load.json_state(json_path)
            if d is None:
                continue

            for i in range(len(d)):
                if 'steps' not in d[i]:
                    continue
                self.detailed_ids.append((ids_idx, i))

        if random_subset_size is not None:
            self.detailed_ids = random.sample(self.detailed_ids, min(len(self.ids), random_subset_size))

        print(len(self.detailed_ids))

    def __len__(self):
        return len(self.detailed_ids)

    def __getitem__(self, idx):

        ids_idx, line_idx = self.detailed_ids[idx]
        gt_json_path, img_path = self.ids[ids_idx]
        gt_json = safe_load.json_state(gt_json_path)

        absolute_steps = []

        if 'steps' not in gt_json[line_idx]:
            return None

        steps = gt_json[line_idx]['steps']

        for step_index, step_data in enumerate(gt_json[line_idx]['steps'][:-1]):
            next_step_data = steps[step_index + 1]
            step_points = step_to_points(step_data)
            next_step_points = step_to_points(next_step_data)
            angle_to_next = angle_between_points(step_points["base_point"], next_step_points["base_point"])

            if step_index <= len(steps) - 2:
                angle_after_that = angle_to_next
            else:
                angle_after_that = angle_between_points(next_step_points["base_point"],
                                                        step_to_points(steps[step_index + 2])["base_point"])
            absolute_steps.append(torch.stack([
                torch.tensor([step_points["upper_point"].x, step_points["upper_point"].y]),
                torch.tensor([step_points["base_point"].x, step_points["base_point"].y]),
                torch.tensor([step_points["lower_point"].x, step_points["lower_point"].y]),
                torch.stack([torch.tensor(angle_to_next), torch.tensor(0)]),
                torch.stack([torch.tensor(step_points["stop_confidence"]), torch.tensor(0)])

                # torch.tensor(angle_after_that)
            ]))

        img = cv2.imread(img_path)
        if self.augmentation:
            img = augmentation.apply_random_color_rotation(img)
            img = augmentation.apply_tensmeyer_brightness(img)

        img = img.astype(np.float32)
        img = img.transpose()
        img = img / 128.0 - 1.0
        img = torch.from_numpy(img)

        gt = gt_json[line_idx]['gt']

        result = {
            "img_path": img_path,
            "img": img,
            "steps": torch.stack(absolute_steps),
            "gt": gt
        }

        return result
