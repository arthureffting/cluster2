import pylab as pl
import os
import cairocffi as cairo
from descartes import PolygonPatch
from shapely.geometry import Point, MultiPolygon

from scripts.new.iam_conversion.alpha_shape import alpha_shape
from scripts.utils.files import create_folders


def plot_polygon(polygon):
    fig = pl.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    margin = .3

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(polygon, fc='#999999', ec='#000000', fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig


def run_transformation_approach(pair, alpha=0.004, visualization_path=None):
    print("Transforming pair #" + str(pair.index))

    image = cairo.ImageSurface.create_from_png(pair.img)
    context = cairo.Context(image)

    for component in pair.get_components():
        context.rectangle(component["x"], component["y"], component["width"], component["height"], )
        context.set_source_rgba(0, 0, 1, 0.1)
        context.fill()
    size_threshold = 50

    line_components = {}
    for component in pair.get_components():
        data = pointsOf(component)
        if data["index"] not in line_components:
            line_components[data["index"]] = []
        line_components[data["index"]].append(data["top_left"])
        line_components[data["index"]].append(data["top_right"])
        line_components[data["index"]].append(data["bottom_right"])
        line_components[data["index"]].append(data["bottom_left"])

    multi = 0

    context.set_operator(cairo.OPERATOR_MULTIPLY)
    context.set_line_width(5)
    context.set_source_rgba(1, 0, 0, 1)

    transformation = pair.get_transformation()

    used_data = {}
    amount_of_lines = len(transformation["lines"])

    for line in transformation["lines"]:
        baseline = line["baseline"]
        start = baseline[0]
        end = baseline[1]
        context.move_to(start[0], start[1])
        context.line_to(end[0], end[1])
        context.stroke()

    for line_index in line_components:
        points = line_components[line_index]

        # The following proves that I don't know numpy in the slightest
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        coords = [Point(p[0], p[1]) for p in points]
        concave_hull, edge_points = alpha_shape(coords, alpha=alpha)

        if isinstance(concave_hull, MultiPolygon):
            continue

        for exterior in [concave_hull.exterior]:
            context.set_operator(cairo.OPERATOR_MULTIPLY)
            context.set_line_width(3)
            context.set_source_rgba(0, 1, 0.3, 1)
            context.move_to(exterior.coords[0][0], exterior.coords[0][1])
            for point in exterior.coords:
                context.line_to(point[0], point[1])
            context.stroke()

        used_data[line_index] = {
            "index": line_index,
            "hull": concave_hull.exterior,
            "baseline": transformation["lines"][line_index]["baseline"],
            "text": transformation["lines"][line_index]["gt"],
        }

    if visualization_path is not None:
        visualization_path = os.path.join(visualization_path, str(pair.index) + ".png")
        create_folders(visualization_path)
        image.write_to_png(visualization_path)
    return used_data


def pointsOf(component):
    x, y, width, height, index = component["x"], \
                                 component["y"], \
                                 component["width"], \
                                 component["height"], \
                                 component["index"]
    return {
        "index": index,
        "top_left": [x, y],
        "top_right": [x + width, y],
        "bottom_right": [x + width, y + height],
        "bottom_left": [x, y + height],
    }
