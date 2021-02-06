from shapely.geometry import Point

from scripts.utils.geometry import *


class TrainingStep:

    def __init__(self, line, step_data=None):
        self.line = line
        self.stop_confidence = 0
        if step_data is not None:
            self.upper_point = Point(step_data["upper_point"][0], step_data["upper_point"][1])
            self.lower_point = Point(step_data["lower_point"][0], step_data["lower_point"][1])
            self.base_point = Point(step_data["base_point"][0], step_data["base_point"][1])
            self.angle = angle_between_points(self.base_point, self.upper_point) + 90.0
            self.upper_size = self.calculate_upper_height()
            self.lower_size = self.calculate_lower_height()

    def calculate_upper_height(self):
        return self.base_point.distance(self.upper_point)

    def calculate_lower_height(self):
        return self.base_point.distance(self.lower_point)

    def index(self):
        for i in range(len(self.line.steps)):
            if self.line.steps[i] is self:
                return i
        return -1

    def next(self, enforce=True):
        if self.index() >= len(self.line.steps) - 1:
            if enforce:
                step = TrainingStep(self.line)
                step.angle = self.angle
                step.upper_size = self.upper_size
                step.lower_size = 0
                step.base_point = get_new_point(self.base_point, self.angle, step.upper_size)
                step.upper_point = get_new_point(step.base_point, step.angle - 90, step.upper_size)
                step.lower_point = get_new_point(step.base_point, step.angle + 90, step.lower_size)
                # self.line.steps.append(step)
                return step
            return None
        else:
            return self.line.steps[self.index() + 1]

    def previous(self):
        if self.index() == 0:
            return None
        else:
            return self.line.steps[self.index() - 1]
