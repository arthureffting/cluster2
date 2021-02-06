import cv2
import numpy as np

from scripts.new.domain.training_step import TrainingStep, angle_between_points, get_new_point


class TrainingLine:

    def __init__(self, image, line_data):
        self.image = image
        self.index = line_data["index"]
        self.text = line_data["text"]
        self.steps = [TrainingStep(self, step) for step in line_data["steps"]]
        self.sol = self.steps[0]

    def baseline(self, extended=True):

        extension = []
        if extended:
            first_angle = angle_between_points(self.steps[0].base_point, self.steps[1].base_point)
            distance = self.steps[0].base_point.distance(self.steps[1].base_point)
            backward_point = get_new_point(self.steps[0].base_point, first_angle + 180, 2 * distance)
            extension.append(backward_point)

        end_extension = []
        if extended:
            new_last_point = get_new_point(self.steps[-1].base_point,
                                           self.steps[-1].angle,
                                           2 * self.steps[-1].upper_size)
            end_extension.append(new_last_point)

        return extension + [step.base_point for step in self.steps] + end_extension

    def hull(self, extended=True):

        forward_points = []
        backward_points = []
        steps = self.steps

        if extended:
            first_angle = angle_between_points(self.steps[0].base_point, self.steps[1].base_point)
            up_backward = get_new_point(self.sol.upper_point, first_angle + 180, 2 * self.sol.calculate_upper_height())
            bottom_backward = get_new_point(self.sol.lower_point, first_angle + 180,
                                            2 * self.sol.calculate_upper_height())
            forward_points.append(up_backward)
            backward_points.append(bottom_backward)

        for step in steps:
            forward_points.append(step.upper_point)
            backward_points.append(step.lower_point)

        if extended:
            up_forward = get_new_point(self.steps[-1].upper_point,
                                       self.steps[-1].angle,
                                       2 * self.steps[-1].upper_size)
            bottom_forward = get_new_point(self.steps[-1].lower_point,
                                           self.steps[-1].angle,
                                           2 * self.steps[-1].upper_size)
            forward_points.append(up_forward)
            backward_points.append(bottom_forward)

        backward_points.reverse()

        forward_points.extend(backward_points)
        return forward_points

    def masked(self):
        image = self.image.raw()
        mask = 255 * np.ones(image.shape, dtype=image.dtype)
        for index, step in enumerate(self.steps[:-1]):
            next = self.steps[index + 1]
            all_points = [step.upper_point, next.upper_point, next.lower_point, step.lower_point]
            all_points = [[p.x, p.y] for p in all_points]
            all_points = np.array(all_points, np.int32)
            cv2.fillConvexPoly(mask, all_points, (0, 0, 0))
        im_thresh_color = cv2.bitwise_or(image, mask)
        return im_thresh_color

    def sampling_range(self):
        return range(len(self.steps) - 2)

    def sampling_steps(self):
        range = self.sampling_range()
        return self.steps[range[0]:range[-1]]
