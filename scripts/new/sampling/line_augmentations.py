from shapely.geometry import LineString

from scripts.new.domain.training_step import TrainingStep
from scripts.utils.geometry import *


class LineAugmentation:

    # Extends lthe line by a number of steps with a size and confidence decay
    @staticmethod
    def extend(line, by=6, size_decay=0.9, confidence_decay=0.825):

        confidences = [1]
        for i in range(by - 1):
            confidences.append(confidences[-1] * confidence_decay)
        confidences.reverse()

        for index in range(by):
            direction = angle_between_points(line.steps[-2].base_point, line.steps[-1].base_point)

            upper_height = line.steps[-1].calculate_upper_height()
            lower_height = line.steps[-1].calculate_lower_height()

            next_base_point = get_new_point(line.steps[-1].base_point, direction, upper_height)
            next_upper_point = get_new_point(next_base_point, direction - 90.0, upper_height * size_decay)
            next_lower_point = get_new_point(next_base_point, direction + 90.0, lower_height * size_decay)

            step_data = {
                "base_point": [next_base_point.x, next_base_point.y],
                "lower_point": [next_lower_point.x, next_lower_point.y],
                "upper_point": [next_upper_point.x, next_upper_point.y],
            }

            step = TrainingStep(line, step_data)
            step.stop_confidence = confidences[index]
            line.steps.append(step)

    # Projects the SOLs backwards
    @staticmethod
    def extend_backwards(line):

        if len(line.steps) < 2:
            return

        direction = angle_between_points(line.steps[0].base_point, line.steps[1].base_point)

        upper_height = line.steps[0].calculate_upper_height()
        lower_height = line.steps[0].calculate_lower_height()

        next_base_point = get_new_point(line.steps[0].base_point, direction - 180, upper_height)
        next_upper_point = get_new_point(next_base_point, direction - 90.0, upper_height)
        next_lower_point = get_new_point(next_base_point, direction + 90.0, lower_height)

        step_data = {
            "base_point": [next_base_point.x, next_base_point.y],
            "lower_point": [next_lower_point.x, next_lower_point.y],
            "upper_point": [next_upper_point.x, next_upper_point.y],
        }

        step = TrainingStep(line, step_data)
        step.stop_confidence = 0.0
        line.steps = [step] + line.steps

    # Assures that the angle of each step points to the baseline point of the next
    @staticmethod
    def normalize(line):
        for step_index in range(len(line.steps) - 1):
            line.steps[step_index].angle = angle_between_points(line.steps[step_index].base_point,
                                                                line.steps[step_index + 1].base_point)

    # If the angle between the first two steps is too big,
    # Cancel first step and replace it with a backward projection of the second
    @staticmethod
    def prevent_wrong_start(line, angle_threshold=45):

        if len(line.steps) < 3:
            return

        first_step = line.steps[0]
        second_step = line.steps[1]
        third_step = line.steps[2]

        first_angle = angle_between_points(first_step.base_point, second_step.base_point)
        second_angle = angle_between_points(second_step.base_point, third_step.base_point)
        # Check second and third
        if abs(second_angle) > angle_threshold:
            line.steps = line.steps[2:]
            LineAugmentation.extend_backwards(line)
            LineAugmentation.extend_backwards(line)
        elif abs(first_angle) > angle_threshold:
            line.steps = line.steps[1:]
            LineAugmentation.extend_backwards(line)

        # Afterwards check if the results has crossings

        first_step = line.steps[0]
        second_step = line.steps[1]
        line_string = LineString([(first_step.base_point.x, first_step.base_point.y),
                                  (first_step.upper_point.x, first_step.upper_point.y),
                                  (second_step.upper_point.x, second_step.upper_point.y),
                                  (second_step.base_point.x, second_step.base_point.y),
                                  (second_step.lower_point.x, second_step.lower_point.y),
                                  (first_step.lower_point.x, first_step.lower_point.y)])
        if not line_string.is_simple:
            print("Correcting intersection at", line.image.path)
            line.steps = line.steps[1:]
            LineAugmentation.extend_backwards(line)

    # Makes sure there are no steps with a height below the minimum,
    # and also no steps too close to each other
    @staticmethod
    def enforce_minimum_height(line, minimum_height=16):

        ignored = []

        for step_index in range(len(line.steps) - 1):

            if line.steps[step_index] in ignored:
                continue

            if step_index < len(line.steps) - 1:
                next_step = line.steps[step_index + 1]
                distance = line.steps[step_index].base_point.distance(next_step.base_point)
                if distance < minimum_height / 2:
                    ignored.append(next_step)

            if line.steps[step_index].calculate_upper_height() < minimum_height:
                new_upper_point = get_new_point(line.steps[step_index].base_point,
                                                line.steps[step_index].angle - 90,
                                                minimum_height)
                line.steps[step_index].upper_point = new_upper_point

        line.steps = [l for l in line.steps if l not in ignored]
