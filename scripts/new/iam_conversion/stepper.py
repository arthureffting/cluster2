import math
import os
from math import sqrt

import cairocffi as cairo
from shapely.geometry import LineString, Point, MultiPoint

from scripts.utils.files import save_to_json, create_folders
from scripts.utils.geometry import angle_between_points, get_new_point




def to_points(array):
    return [Point(p[0], p[1]) for p in array]


def perpendicular(point_a, baseline):
    b = point_a
    a = baseline[1]
    cd_length = 800

    ab = LineString([a, b])
    left = ab.parallel_offset(cd_length / 2, 'left')
    right = ab.parallel_offset(cd_length / 2, 'right')
    c = left.boundary[1]
    d = right.boundary[0]  # note the different orientation for right offset
    cd = LineString([c, d])
    return cd

def slope(baseline):
    dy = baseline[1][1] - baseline[0][1]
    dx = baseline[1][0] - baseline[0][0]
    return dy / dx

def walk(start, slope, amount):
    new_x = start[0] + amount
    new_y = start[1] + amount * slope
    return [new_x, new_y]


def distance(p1, p2):
    return sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))


def to_steps(data, pairs, visualize=True):
    result = {
        "images": []
    }

    for i, page_index in enumerate(data):
        pair = pairs[i]

        print("Stepping pair #" + str(pair.index))

        image_data = {
            "index": pair.index,
            "filename": pair.img,
            "lines": []
        }

        image = cairo.ImageSurface.create_from_png(pair.img)
        context = cairo.Context(image)

        if visualize:
            for component in pair.get_components():
                context.rectangle(component["x"], component["y"], component["width"], component["height"], )
                context.set_source_rgba(0, 0, 1, 0.1)
                context.fill()

        for line_index in data[page_index]:

            line = data[page_index][line_index]
            baseline = line["baseline"]
            hull = line["hull"]
            line_data = {
                "text": line["text"],
                "steps": []
            }

            line_slope = slope(baseline)
            start_point = baseline[0]
            distance_walked = 0
            total_distance = distance(baseline[0], baseline[1])

            height_threshold = 20
            context.set_operator(cairo.OPERATOR_MULTIPLY)
            context.set_line_width(5)
            upper_points = []
            lower_points = []
            baseline_points = []

            while distance_walked < total_distance:
                intersecting_line = perpendicular(start_point, baseline)
                intersection = intersecting_line.intersection(hull)

                upper_point = None
                lower_point = None

                if isinstance(intersection, MultiPoint) and len(intersection.bounds) == 4:
                    upper_point = [intersection.bounds[0], intersection.bounds[1]]
                    lower_point = [intersection.bounds[2], intersection.bounds[3]]
                elif isinstance(intersection, LineString) and len(intersection.bounds) == 4:
                    upper_point = [intersection.bounds[0], intersection.bounds[1]]
                    lower_point = [intersection.bounds[2], intersection.bounds[3]]
                elif isinstance(intersection, Point):
                    print("Intersection was point, moving forward")
                    start_point = walk(start_point, line_slope, 4)
                    continue
                else:
                    if distance_walked == 0:
                        start_point = walk(start_point, line_slope, 4)
                    else:
                        print("No intersection, skipping line " + str(line_index) + " of " + str(
                            pair.index) + " after walking" + str(distance_walked))
                        distance_walked = total_distance
                    continue

                if upper_point is not None and lower_point is not None:

                    upper_points.append(upper_point)
                    lower_points.append(lower_point)

                    baseline_intersection = LineString(
                        [Point(upper_point[0], upper_point[1]), Point(lower_point[0], lower_point[1])]) \
                        .intersection(LineString(to_points(baseline)))

                    baseline_point = None
                    if isinstance(baseline_intersection, Point) and len(baseline_intersection.bounds) > 1:
                        baseline_point = [baseline_intersection.bounds[0], baseline_intersection.bounds[1]]
                    else:
                        baseline_point = lower_point
                    baseline_points.append(baseline_point)

                    height = distance(upper_point, baseline_point)

                    if height < height_threshold and distance_walked == 0:
                        # The first point doesnt have a height
                        if distance_walked == 0:
                            angle = angle_between_points(to_points(baseline)[0], to_points(baseline)[1])
                            new_upper_point = get_new_point(to_points(baseline)[0], angle - 90, height_threshold)
                            upper_point = [new_upper_point.x, new_upper_point.y]

                    if height < height_threshold:
                        height = height_threshold

                    context.set_source_rgba(1, 0, 1, 1)
                    context.move_to(upper_point[0], upper_point[1])
                    context.line_to(lower_point[0], lower_point[1])
                    context.stroke()

                    context.set_source_rgba(0, 0, 1, 0.1)
                    context.move_to(start_point[0], start_point[1])
                    start_point = walk(start_point, line_slope, height)
                    distance_walked += height
                    context.line_to(start_point[0], start_point[1])
                    context.stroke()

                else:
                    distance_walked = total_distance

            for pc in [baseline_points, upper_points, lower_points]:
                if len(pc) == 0:
                    continue
                context.set_source_rgba(1, 0, 1, 0.3)
                context.move_to(pc[0][0], pc[0][1])
                for bp in pc:
                    context.line_to(bp[0], bp[1])
                context.stroke()

            for i in range(len(baseline_points)):
                line_data["steps"].append({
                    "upper_point": upper_points[i],
                    "lower_point": lower_points[i],
                    "base_point": baseline_points[i],
                })

            line_data["index"] = line_index
            image_data["lines"].append(line_data)

        result["images"].append(image_data)
        save_path = os.path.join(pair.base, "json", str(image_data["index"]) + ".json")
        save_to_json(image_data, save_path)
        if visualize:
            visualization_path = os.path.join(pair.base, "stepped", str(pair.index) + ".png")
            create_folders(visualization_path)
            image.write_to_png(visualization_path)

    return result
