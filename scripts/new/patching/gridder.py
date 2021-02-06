import numpy as np
import torch
from torch.autograd import Variable

from scripts.new.patching import transformation_utils
from scripts.new.patching.fast_patch_view import get_patches
from scripts.new.patching.gridgen import GridGen


class Gridder:
    def __init__(self, grid_size=32):
        self.output_grid_size = grid_size

    def get_grid(self, image, inputs):
        batch_size = image.size(0)
        renorm_matrix = transformation_utils.compute_renorm_matrix(image)
        expanded_renorm_matrix = renorm_matrix.expand(batch_size, 3, 3)
        t = ((np.arange(self.output_grid_size) + 0.5) / float(self.output_grid_size))[:, None].astype(np.float32)
        t = np.repeat(t, axis=1, repeats=self.output_grid_size)
        t = Variable(torch.from_numpy(t), requires_grad=False).cuda()
        s = t.t()
        t = t[:, :, None]
        s = s[:, :, None]
        interpolations = torch.cat([
            (1 - t) * s,
            (1 - t) * (1 - s),
            t * s,
            t * (1 - s),
        ], dim=-1)
        view_window = Variable(torch.cuda.FloatTensor([
            [2, 0, 2],
            [0, 2, 0],
            [0, 0, 1]
        ])).expand(batch_size, 3, 3)
        step_bias = Variable(torch.cuda.FloatTensor([
            [1, 0, 2],
            [0, 1, 0],
            [0, 0, 1]
        ])).expand(batch_size, 3, 3)
        invert = Variable(torch.cuda.FloatTensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])).expand(batch_size, 3, 3)
        grid_gen = GridGen(self.output_grid_size, self.output_grid_size)
        current_window = transformation_utils.get_init_matrix(inputs)
        crop_window = current_window.bmm(view_window)
        resampled = get_patches(image, crop_window, grid_gen, False)
        return resampled
