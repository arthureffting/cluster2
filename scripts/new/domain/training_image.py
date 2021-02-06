import os

import cv2
import numpy
import numpy as np
import torch
from torch.autograd import Variable

from scripts.new.domain.training_line import TrainingLine
from scripts.utils.files import create_folders
from scripts.utils.image_transposal import transpose_to_torch


class TrainingImage:
    rescale_range = [384, 640]

    def __init__(self, image_data):
        self.index = image_data["index"]
        self.path = image_data["filename"]
        self.lines = [TrainingLine(self, line_data) for line_data in image_data["lines"]]

    def transpose(self):
        return transpose_to_torch(cv2.imread(self.path))

    def get_sols_gt(self, dtype=torch.cuda.FloatTensor):
        sol_gt = np.zeros((1, len(self.lines), 5), dtype=np.float32)

        for j, line in enumerate(self.lines):
            sol_gt[:, j, 0] = 1.0
            sol_gt[:, j, 1] = line.sol.upper_point.x
            sol_gt[:, j, 2] = line.sol.upper_point.y
            sol_gt[:, j, 3] = line.sol.lower_point.x
            sol_gt[:, j, 4] = line.sol.lower_point.y

        sol_gt = torch.from_numpy(sol_gt)
        return Variable(sol_gt.type(dtype), requires_grad=False)

    def reversed(self):
        return self.raw()[:, ::-1]

    def raw(self):
        return cv2.imread(self.path)[..., ::-1]

    def masked(self):
        image = self.raw()
        mask = 255 * np.ones(image.shape, dtype=image.dtype)
        for line in self.lines:
            for index, step in enumerate(line.steps[:-1]):
                next = line.steps[index + 1]
                all_points = [step.upper_point, next.upper_point, next.lower_point, step.lower_point]
                all_points = [[p.x, p.y] for p in all_points]
                all_points = numpy.array(all_points, numpy.int32)
                cv2.fillConvexPoly(mask, all_points, (0, 0, 0))
        im_thresh_color = cv2.bitwise_or(image, mask)
        return im_thresh_color

    def folder(self, folder_name):
        path = os.path.join(folder_name, str(self.index) + ".png")
        create_folders(path)
        return path
