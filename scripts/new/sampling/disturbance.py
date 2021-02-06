import random

import numpy as np
from shapely.geometry import Point

"""
Class used for adding disturbances to training samples
"""


class Disturbance:

    def __init__(self, parameters):
        self.parameters = parameters
        self.origin_scale = random.uniform(0, 1)  # Random scale for the origin projection
        self.origin_angle = random.uniform(0, 2 * np.pi)  # Random angle for projecting origin
        self.size_distortion = self.normal_sample(sigma=self.parameters.size_distortion, center=1.0)
        self.angle_distortion = self.normal_sample(sigma=self.parameters.angle_distortion, center=0.0)

    # Generates a normal distribution at the center with sigma,
    # and caps it so that nothing
    # beyond center +- sigma exists
    def normal_sample(self, sigma, center=0.0, cap=True):
        result = random.choice(np.random.normal(center, sigma, 1000))
        if cap:
            result = center - sigma if result < center - sigma else (
                center + sigma if result > center + sigma else result)
        return result

    # Applies the current distortion to the origin, size and angle of a sampling window
    def apply(self, origin, size, angle):
        scaled_origin_size = size * self.origin_scale * self.parameters.origin_distortion
        origin_x_distortion = np.cos(self.origin_angle) * scaled_origin_size
        origin_y_distortion = np.sin(self.origin_angle) * scaled_origin_size
        origin = Point(origin.x + origin_x_distortion, origin.y + origin_y_distortion)
        size *= self.size_distortion
        angle += self.angle_distortion
        return origin, size, angle
