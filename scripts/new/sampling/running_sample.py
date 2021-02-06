import cv2
import numpy as np
import torch
from shapely.geometry import Point
from domain.running_line import RunningLine
from domain.running_step import RunningStep
from sampling.viewing_window import ViewingWindow
from utils.image_transposal import transpose_to_torch, rescale


class RunningSample:

    def __init__(self, parameters, image, window):
        self.line = RunningLine(image)
        self.parameters = parameters
        self.image = image
        self.window = window
        self.windows = [window]
        self.predicted_steps = []

    def next(self, output):
        relative_upper = self.window.absolute(Point(output[0].item(), output[1].item()))
        relative_base = self.window.absolute(Point(output[2].item(), output[3].item()))
        relative_lower = self.window.absolute(Point(output[4].item(), output[5].item()))
        stop_confidence = output[7].item()
        next_angle = output[6].item() + self.window.angle

        step = RunningStep(relative_upper,
                           relative_base,
                           relative_lower,
                           stop_confidence,
                           next_angle)

        new_window = ViewingWindow(self.parameters,
                                   self.image,
                                   step.base_point,
                                   max(self.parameters.min_step_size, step.calculate_upper_height()),
                                   step.angle)
        self.window = new_window
        self.predicted_steps.append(step)
        self.windows.append(new_window)
        if len(self.windows) > 5:
            self.windows = self.windows[-5:]
        return step

    def input(self):
        # How much less input available there is than expected
        if len(self.predicted_steps) > 1 and self.predicted_steps[-1].stop_confidence > self.parameters.stop_threshold:
            return None
        try:
            length = min(len(self.windows), self.parameters.sequence_size)
            size = self.parameters.patch_size
            inputs = np.zeros((1, length, 3, size, size), dtype=np.float32)
            src_img = cv2.imread(self.image.image_path)[:, ::-1]
            images = [w.image(src_img) for w in self.windows]
            images = rescale(images, self.parameters.patch_size)
            for index, image in enumerate(images):
                inputs[0, index, :, :, :] = transpose_to_torch(image)
            return torch.from_numpy(inputs)
        except Exception as e:
            print(e)
            return None
