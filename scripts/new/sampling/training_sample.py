import random

import cv2
import numpy as np
import torch
from shapely.geometry import Point

from sampling.disturbance import Disturbance
from sampling.viewing_window import ViewingWindow
from utils.geometry import get_new_point, angle_between_points
from utils.image_handling import subimage
from utils.image_transposal import transpose_to_torch, apply_random_color_rotation, apply_tensmeyer_brightness, rescale


class TrainingSample:

    def __init__(self, parameters, line, previous_steps, next_steps, step_beyond=None):
        self.parameters = parameters
        self.line = line
        self.run_length = len(next_steps)
        self.previous_steps = previous_steps
        self.next_steps = next_steps
        self.step_beyond = step_beyond
        self.window = None

    def input_images(self, disturbance=None, augment=True):
        # missing_windows = self.parameters.sequence_size - len(self.previous_steps)

        length = min(len(self.previous_steps), self.parameters.sequence_size)
        size = self.parameters.patch_size
        inputs = np.zeros((1, length, 3, size, size), dtype=np.float32)

        images = []

        # Fill out zeros when the length is less than the sequence size
        # for i in range(missing_windows):
        #    images.append(np.zeros(shape=(self.parameters.patch_size, self.parameters.patch_size, 3)))

        used_steps = self.previous_steps[-length:]

        source_image = cv2.imread(self.line.image.image_path)[:, ::-1]

        for index, step in enumerate(used_steps):
            angle = step.angle
            size = step.calculate_upper_height()
            origin = step.base_point

            if disturbance is not None and index == len(used_steps) - 1:
                origin, size, angle = disturbance.apply(origin, size, angle)

            self.window = ViewingWindow(self.parameters, self.line.image, origin, size, angle)
            images.append(self.window.image(source_image))

        images = rescale(images, self.parameters.patch_size)

        if augment:
            images = apply_random_color_rotation(images)
            images = apply_tensmeyer_brightness(images)

        # Add to batch
        for index, image in enumerate(images):
            inputs[0, index, :, :, :] = transpose_to_torch(image)

        return inputs

    def input(self, disturb=True, augment=True):
        try:
            return torch.from_numpy(
                self.input_images(disturbance=Disturbance(self.parameters) if disturb else None, augment=augment))
        except:
            return None

    def desired_output(self):
        # Start the viewing window at the last previous step
        previous_step = self.previous_steps[-1]
        target_step = self.next_steps[0]
        beyond_step = self.step_beyond

        # Calculate desired output based on current viewing window
        # and current target step
        desired_angle = angle_between_points(target_step.base_point, beyond_step.base_point) \
            if beyond_step is not None else angle_between_points(previous_step.base_point, target_step.base_point)
        desired_confidence = target_step.stop_confidence

        desired_upper_output = self.window.relative(target_step.upper_point)
        desired_base_output = self.window.relative(target_step.base_point)
        desired_lower_output = self.window.relative(target_step.lower_point)

        return torch.tensor([desired_upper_output.x,
                             desired_upper_output.y,
                             desired_base_output.x,
                             desired_base_output.y,
                             desired_lower_output.x,
                             desired_lower_output.y,
                             desired_angle - self.window.angle,
                             desired_confidence], dtype=torch.float32)
