import cairocffi as cairo
import math

from scripts.utils.files import create_folders
from scripts.utils.geometry import get_new_point, angle_between_points, midpoint, viewing_window_points


class Painter:

    def __init__(self, image=None, path=None):
        self.image = image
        self.surface = cairo.ImageSurface.create_from_png(image.path if image is not None else path)
        self.context = cairo.Context(self.surface)

    def save(self, path="test.png"):
        create_folders(path)
        self.surface.write_to_png(path)

    def draw_outlines(self):
        for line in self.image.lines:
            self.draw_line([step.base_point for step in line.steps], color=(0, 0, 1), line_width=2)
            for step_index, step in enumerate(line.steps[:-1]):
                next_step = line.steps[step_index + 1]
                self.draw_line([step.lower_point, step.upper_point], color=(1, 0, 1, 0.5), line_width=1)
                self.draw_area(
                    [step.upper_point, next_step.upper_point, next_step.lower_point, step.lower_point],
                    fill_color=(1.25 * next_step.stop_confidence, 0, 0, 0.75 * next_step.stop_confidence),
                    line_color=(1, 0, 1),
                    line_width=2)

    def draw_line(self, points,
                  color=(1, 0, 0, 1),
                  line_width=1,
                  joints=False,
                  joint_color=(1, 0, 0, 1),
                  joint_radius=4):
        if len(points) < 2:
            return
        self.context.set_line_width(line_width)
        self.context.set_source_rgba(*color)
        self.context.move_to(points[0].x, points[0].y)
        for i in range(1, len(points)):
            self.context.line_to(points[i].x, points[i].y)
        self.context.stroke()

        if joints:
            for point in points:
                self.draw_point(point, color=joint_color, radius=joint_radius)

    def draw_area(self, points, line_color=(1, 0, 0, 1), fill_color=(1, 0, 0, 1), line_width=0):
        if len(points) < 3:
            return
        self.context.set_line_width(line_width)
        self.context.move_to(points[0].x, points[0].y)
        for i in range(1, len(points)):
            self.context.line_to(points[i].x, points[i].y)
        self.context.close_path()
        self.context.set_source_rgba(*fill_color)
        self.context.fill_preserve()
        self.context.set_source_rgba(*line_color)
        self.context.stroke()

    def draw_point(self, point, color=(1, 0, 0, 1.0), radius=2):
        self.context.move_to(point.x, point.y)
        self.context.arc(point.x, point.y, radius, 0, 2 * math.pi)
        self.context.close_path()
        self.context.set_source_rgba(*color)
        self.context.fill()

    def draw_reach(self, b, target_point, range=(0, 360), color=(1, 1, 0, 0.5)):
        base_point = midpoint(b, target_point)
        angle = angle_between_points(base_point, target_point)
        self.context.move_to(base_point.x, base_point.y)
        self.context.arc(base_point.x, base_point.y, base_point.distance(target_point), math.radians(angle + range[0]),
                         math.radians(angle + range[1]))
        self.context.close_path()
        self.context.set_source_rgba(*color)
        self.context.fill()

    def draw_step(self, step, color=(0, 0, 1, 1)):
        self.draw_point(step.upper_point, color=color)
        self.draw_point(step.base_point, color=color)
        self.draw_point(step.lower_point, color=color)
        self.draw_line([step.upper_point, step.lower_point], color=color)

    def draw_ground_truth(self, steps, stop_threshold=1.1):

        steps_filtered = []

        for step in steps:
            if step.stop_confidence <= stop_threshold:
                steps_filtered.append(step)
            else:
                break

        for step_index, step in enumerate(steps_filtered[:-1]):
            next = steps_filtered[step_index + 1]
            self.draw_area([step.upper_point, next.upper_point, next.base_point, next.lower_point, step.lower_point,
                            step.base_point],
                           fill_color=(1, 0, 0, next.stop_confidence),
                           line_color=(1, 0, 1, 1),
                           line_width=2)
            self.draw_line([step.base_point, next.base_point], color=(0, 0, 1, 1), line_width=2)

    def draw_viewing_window(self, step, ratio, line_color=(1, 0, 1, 1), fill_color=(1, 0, 1, 0.15), line_width=3):
        focus = get_new_point(step.base_point,
                              step.angle,
                              ratio * step.calculate_upper_height() / 2)
        points = viewing_window_points(focus, step.calculate_upper_height(), step.angle, ratio=ratio)
        self.draw_area(points, fill_color=fill_color, line_color=line_color, line_width=line_width)
