import math

import cv2
import numpy as np

from scripts.utils.geometry import get_new_point, angle_between_points
from scripts.utils.image_handling import hconcat_resize_min, vconcat_resize_min


class Dewarper2:

    def __init__(self, steps):
        self.steps = steps

    def is_point_out_of_bounds(self, img, point):
        return point.x < 0 or point.y < 0 or point.x > img.shape[1] or point.y > img.shape[0]

    def is_out_of_bounds(self, img, step):
        return self.is_point_out_of_bounds(img, step.upper_point) \
               or self.is_point_out_of_bounds(img, step.base_point) \
               or self.is_point_out_of_bounds(img, step.lower_point)

    def valid_steps(self, img):
        return [step for step in self.steps if not self.is_out_of_bounds(img, step)]

    def dewarped(self, img):
        steps = self.valid_steps(img)
        max_upper_height = max([step.calculate_upper_height() for step in self.steps])
        max_lower_height = max([step.calculate_lower_height() for step in self.steps])
        total_height = max_upper_height + max_lower_height
        extraction_points = []

        concatenated_img = None

        for index, current_step in enumerate(self.steps[:-1]):
            next_step = self.steps[index + 1]
            angle = angle_between_points(current_step.base_point, next_step.base_point)
            upper = get_new_point(current_step.base_point, angle - 90, max_upper_height)
            lower = get_new_point(current_step.base_point, angle + 90, max_lower_height)
            width = current_step.base_point.distance(next_step.base_point)
            extraction_points.append([upper, lower, width])

        for index, extraction_point in enumerate(extraction_points[:-1]):
            [upper_left, lower_left, width] = extraction_point
            [upper_right, lower_right, _] = extraction_points[index + 1]
            background = 255 * np.ones((int(total_height), int(width), 3), np.uint8)

            lower_src = np.array([[upper_left.x, upper_left.y],
                                  [upper_right.x, upper_right.y],
                                  [lower_right.x, lower_right.y],
                                  [lower_left.x, lower_left.y]])
            # Destination rectangles
            lower_dst = np.array([[0, 0],
                                  [width, 0.0],
                                  [width, max_upper_height + max_lower_height],
                                  [0.0, max_upper_height + max_lower_height]])

            lower_perspective, _ = cv2.findHomography(lower_src, lower_dst)
            warped_image = cv2.warpPerspective(img,
                                               lower_perspective,
                                               (background.shape[1], background.shape[0]))
            concatenated_img = warped_image if concatenated_img is None else hconcat_resize_min(
                [concatenated_img, warped_image])

        return concatenated_img


class Dewarper:

    def __init__(self, steps):
        self.steps = steps

    def shortAngleDist(self, a0, a1):
        max = math.pi * 2
        da = (a1 - a0) % max
        return 2 * da % max - da

    def angleLerp(self, a0, a1, t):
        two_pi = math.pi * 2
        a0 = (math.radians(a0) + two_pi) % two_pi
        a1 = (math.radians(a1) + two_pi) % two_pi
        return math.degrees(a0 + self.shortAngleDist(a0, a1) * t)

    def dewarped(self, img) -> object:

        if img is None:
            print("[DEWARP FAIL]", "Image is none")
            return None

        try:
            downc = self.lower_concat(img)
        except Exception as e:
            print("\n[LOWER FAIL]", e)
            downc = None

        try:
            upc = self.upper_concat(img)
        except Exception as e:
            print("\n[UPPER FAIL]", e)
            return None

        del img
        return upc if downc is None else vconcat_resize_min([upc, downc])

    def is_out_of_bounds(self, img, step):
        return self.is_point_out_of_bounds(img, step.upper_point) \
               or self.is_point_out_of_bounds(img, step.base_point) \
               or self.is_point_out_of_bounds(img, step.lower_point)

    def is_point_out_of_bounds(self, img, point):
        return point.x < 0 or point.y < 0 or point.x > img.shape[1] or point.y > img.shape[0]

    def valid_steps(self, img):
        return [step for step in self.steps if not self.is_out_of_bounds(img, step)]

    def lower_concat(self, img):

        lower_image = None

        steps = self.valid_steps(img)
        box_height = max([step.calculate_lower_height() for step in steps])

        if int(box_height) == 0:
            return None

        for step_index, step in enumerate(steps[:-1]):
            next_step = steps[step_index + 1]
            width = step.base_point.distance(next_step.base_point)
            left_height = step.calculate_lower_height()
            right_height = next_step.calculate_lower_height()

            lower_background = 255 * np.ones((int(box_height), int(width), 3), np.uint8)
            try:
                # Trapezoids
                lower_src = np.array([[step.base_point.x, step.base_point.y],
                                      [next_step.base_point.x, next_step.base_point.y],
                                      [next_step.lower_point.x, next_step.lower_point.y],
                                      [step.lower_point.x, step.lower_point.y]])
                # Destination rectangles
                lower_dst = np.array([[0, 0],
                                      [width, 0.0],
                                      [width, right_height],
                                      [0.0, left_height]])

                lower_perspective, _ = cv2.findHomography(lower_src, lower_dst)
                lower_out = cv2.warpPerspective(img,
                                                lower_perspective,
                                                (lower_background.shape[1], lower_background.shape[0]))
                lower_image = lower_out if lower_image is None else hconcat_resize_min([lower_image, lower_out])
            except Exception as e:
                lower_background = 255 * np.ones((int(box_height), int(width), 3), np.uint8)
                lower_image = lower_background if lower_image is None else hconcat_resize_min(
                    [lower_image, lower_background])

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        return lower_image

    def upper_concat(self, img):

        upper_image = None

        steps = self.valid_steps(img)
        box_height = max([step.calculate_upper_height() for step in steps])

        for step in steps:
            if step.calculate_upper_height() < 16:
                step.upper_point = get_new_point(step.base_point,
                                                 step.angle - 90,
                                                 16)

        for step_index, step in enumerate(steps[:-1]):
            next_step = steps[step_index + 1]
            angle = angle_between_points(step.base_point, next_step.base_point)

            width = step.base_point.distance(next_step.base_point)

            left_upper_height = step.calculate_upper_height()
            right_upper_height = next_step.calculate_upper_height()

            # Trapezoids
            upper_src = np.array([[step.upper_point.x, step.upper_point.y],
                                  [next_step.upper_point.x, next_step.upper_point.y],
                                  [next_step.base_point.x, next_step.base_point.y],
                                  [step.base_point.x, step.base_point.y]])
            # Destination rectangles
            upper_dst = np.array([[0, box_height - left_upper_height],
                                  [width, box_height - right_upper_height],
                                  [width, box_height],
                                  [0.0, box_height]])

            # White background
            upper_background = np.ones((int(box_height), int(width), 3), np.uint8)

            upper_perspective, _ = cv2.findHomography(upper_src, upper_dst)
            upper_out = cv2.warpPerspective(img,
                                            upper_perspective,
                                            (upper_background.shape[1], upper_background.shape[0]))
            upper_image = upper_out if upper_image is None else hconcat_resize_min([upper_image, upper_out])

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        return upper_image

    def section(self):
        interpolated_baseline = []

        ending_angle = self.steps[0].angle - 90

        for step_index, step in enumerate(self.steps[:-1]):
            next_step = self.steps[step_index + 1]

            this_angle = (step.angle - 90)
            next_angle = (next_step.angle - 90)

            starting_angle = ending_angle
            middle_angle = this_angle
            ending_angle = self.angleLerp(this_angle, next_angle, 0.5)

            # Add first point itself
            # interpolated_baseline.append(step.base_point)

            angle = angle_between_points(step.base_point, next_step.base_point)
            distance = step.base_point.distance(next_step.base_point)
            walked = 0
            step_size = 1
            while walked < distance:
                percent_walked = walked / distance

                if percent_walked < 0.5:
                    intersection_angle = self.angleLerp(starting_angle, middle_angle, percent_walked * 2)
                else:
                    assert 0 <= (percent_walked - 0.5) * 2 <= 1
                    intersection_angle = self.angleLerp(middle_angle, ending_angle, (percent_walked - 0.5) * 2)

                baseline_point = get_new_point(step.base_point, angle, walked)
                upper_point = get_new_point(baseline_point, intersection_angle, 80)
                lower_point = get_new_point(baseline_point, intersection_angle - 180, 20)

                interpolated_baseline.append([upper_point, baseline_point, lower_point])
                walked += step_size

        return interpolated_baseline
