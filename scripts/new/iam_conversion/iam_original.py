import json
import math
import os
import sys
from random import random
import cairocffi as cairo
import numpy as np



# Draws the original IAM components on the image
from scripts.utils.files import create_folders


def store_iam_original(pair):
    visualization_path = os.path.join(pair.base, "original", str(pair.index) + ".png")
    create_folders(visualization_path)
    image = cairo.ImageSurface.create_from_png(pair.img)
    context = cairo.Context(image)
    for component in pair.get_components():
        context.rectangle(component["x"], component["y"], component["width"], component["height"], )
        context.set_source_rgba(0, 0, 1, 0.3)
        context.fill()
    image.write_to_png(visualization_path)


def next_color():
    return random(), random(), random()


def draw_output_point(context, sol, color):
    sol_p1 = [sol["x0"], sol["y0"]]
    sol_p2 = [sol["x1"], sol["y1"]]
    mid_point = [(sol_p1[0] + sol_p2[0]) / 2, (sol_p1[1] + sol_p2[1]) / 2]

    # Draw generated baseline
    context.set_operator(cairo.Operator.MULTIPLY)
    context.set_line_width(1)
    context.set_source_rgba(*color, 1)
    context.move_to(sol_p1[0], sol_p1[1])
    context.line_to(sol_p2[0], sol_p2[1])
    context.stroke()

    for point in [sol_p1, sol_p2, mid_point]:
        context.set_operator(cairo.Operator.MULTIPLY)
        context.move_to(point[0], point[1])
        context.arc(point[0], point[1], 3, 0, 2 * math.pi)
        context.close_path()
        context.set_source_rgba(*color, 1)
        context.fill()


def draw_lf_steps(context, lf, color):
    for lf_step in lf:
        draw_output_point(context, lf_step, color)

    p1s = []
    p2s = []

    for lf_step in lf:
        p1 = [lf_step["x0"], lf_step["y0"]]
        p2 = [lf_step["x1"], lf_step["y1"]]
        p1s.append(p1)
        p2s.append(p2)

    context.move_to(p2s[0][0], p2s[0][1])

    for p1 in p1s:
        context.line_to(p1[0], p1[1])

    p2s.reverse()

    for p2 in p2s:
        context.line_to(p2[0], p2[1])

    context.line_to(p1s[0][0], p1s[0][1])

    context.set_operator(cairo.Operator.OVER)
    context.set_line_width(3)
    context.set_source_rgba(*color, 0.03)
    context.fill()


# Saves a visualization of the standard page XML
def store_page_visualization(pair):
    visualization_path = os.path.join(pair.base, "pages", "visualization", pair.stem(extension=".png"))
    create_folders(visualization_path)

    image = cairo.ImageSurface.create_from_png(pair.img)
    context = cairo.Context(image)

    page_data = pair.extract_ground_truth()

    for line in page_data:
        color = next_color()

        # Draw bounding polygon
        polygon = line["bounding_poly"]
        context.move_to(polygon[-1][0], polygon[-1][1])
        for coordinate in polygon:
            context.line_to(*coordinate)
        context.line_to(*(polygon[0]))
        context.set_operator(cairo.Operator.MULTIPLY)
        context.set_line_width(3)
        context.set_source_rgba(*color, 0.2)
        context.stroke()

        # Draw baseline
        baseline = line["baseline"]

        context.move_to(*baseline[0])

        for coordinate in baseline[1:]:
            context.line_to(*coordinate)
            context.set_operator(cairo.Operator.MULTIPLY)
            context.set_line_width(3)
            context.set_source_rgba(*next_color(), 0.7)
            context.stroke()
            context.move_to(*coordinate)

    image.write_to_png(visualization_path)


# Saves a visualization of the standard page XML
def get_custom_transformation(pair, trim_out_of_bounds=True):
    visualization_path = os.path.join(pair.base, "custom_transformation", "visualization", pair.stem(extension=".png"))
    create_folders(visualization_path)

    image = cairo.ImageSurface.create_from_png(pair.img)
    image = convert_to_grayscale(image)
    context = cairo.Context(image)

    page_data = pair.extract_ground_truth()
    lines = []
    for i, line in enumerate(page_data):

        steps = []
        for j, lf in enumerate(line["lf"]):
            step = Step.get_from(lf)
            if step.is_within(image):
                steps.append(step)

        for k, step in enumerate(steps):
            color = (255, 0, 0) if k == 0 else ((0, 0, 255) if k == len(steps) - 1 else (0, 255, 0))
            draw_custom_lf(context, step, color)

        sol = steps[0]
        eol = steps[-1]

        lines.append({
            "steps": list(map(lambda x: x.__dict__, steps)),
            "text": line["ground_truth"]
        })

    image.write_to_png(visualization_path)

    return {
        "origin": pair.base,
        "basename": pair.stem(),
        "lines": lines,
        "image": pair.img,
        "xml": pair.transformation_xml_path,
    }


class Step:

    @staticmethod
    def get_from(data):
        return Step(data["x0"], data["x1"], data["y0"], data["y1"])

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, )

    def __init__(self, x0, x1, y0, y1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

    def is_within(self, image):
        return 0 < self.x0 < image.get_width() \
               and 0 < self.x1 < image.get_width() \
               and 0 < self.y0 < image.get_height() \
               and 0 < self.y1 < image.get_height()


def draw_custom_lf(context, step, color):
    sol_p1 = [step.x0, step.y0]
    sol_p2 = [step.x1, step.y1]
    mid_point = [(step.x0 + step.x1) / 2, (step.y0 + step.y1) / 2]

    # Draw generated baseline
    context.set_operator(cairo.Operator.MULTIPLY)
    context.set_line_width(1)
    context.set_source_rgba(*color, 1)
    context.move_to(sol_p1[0], sol_p1[1])
    context.line_to(sol_p2[0], sol_p2[1])
    context.stroke()

    for point in [sol_p1, sol_p2, mid_point]:
        context.set_operator(cairo.Operator.MULTIPLY)
        context.move_to(point[0], point[1])
        context.arc(point[0], point[1], 3, 0, 2 * math.pi)
        context.close_path()
        context.set_source_rgba(*color, 0.5)
        context.fill()


def convert_to_grayscale(img_in):
    """Convert an image to grayscale.

    Arguments:
        img_in: (cairo.ImageSurface) input image.

    Return:
        (cairo.ImageSurface) image in grayscale, in ARGB32 mode.

    Timing:
        ~100ms to convert an image of 800x800

    Examples:
        # returns a B&W image
        >>> convert_to_grayscale(cairo.ImageSurface.create_from_png('test.png'))
    """
    a = np.frombuffer(img_in.get_data(), np.uint8)
    w, h = img_in.get_width(), img_in.get_height()
    a.shape = (w, h, 4)

    assert sys.byteorder == 'little', (
        'The luminosity vector needs to be switched if we\'re in a big endian architecture. '
        'The alpha channel will be at position 0 instead of 3.')
    alpha = a[:, :, 3]
    alpha.shape = (w, h, 1)

    luminosity_float = np.sum(a * np.array([.114, .587, .299, 0]), axis=2)
    luminosity_int = np.array(luminosity_float, dtype=np.uint8)
    luminosity_int.shape = (w, h, 1)
    grayscale_gbra = np.concatenate((luminosity_int, luminosity_int, luminosity_int, alpha),
                                    axis=2)
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    assert stride == 4 * w, 'We need to modify the numpy code if the stride is different'
    img_out = cairo.ImageSurface.create_for_data(grayscale_gbra, cairo.FORMAT_ARGB32, w, h, stride)

    return img_out
