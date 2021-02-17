import io
import os
from pathlib import Path

import cv2
from shapely.geometry import Point, MultiPolygon, MultiPoint, LineString, Polygon
from skimage import io as ski
from bs4 import BeautifulSoup as bs

from scripts.new.domain.training_image import TrainingImage
from scripts.new.iam_conversion.alpha_shape import alpha_shape
from scripts.new.iam_conversion.stepper import slope, distance, perpendicular, walk, to_points
from scripts.new.sampling.line_augmentations import LineAugmentation
from scripts.utils.files import create_folders, save_to_json
from scripts.utils.geometry import angle_between_points, get_new_point
from scripts.utils.line_dewarper import Dewarper2
from scripts.utils.painter import Painter


class ReadConverter:

    @staticmethod
    def convert(pair, folder_path=os.path.join("data", "sfrs", "read", "pages")):
        img_path, xml_path = pair[0], pair[1]
        stem = Path(img_path).stem
        with open(xml_path, 'r', encoding='utf-8') as myFile:
            b = bs(myFile.read(), "xml")
            xml_regions = [XmlTextRegion(index, img_path, data) for index, data in enumerate(b.findAll("TextRegion"))]

            destination_folder = os.path.join("data", "sfrs", "read")
            for id, region in enumerate(xml_regions):
                # Save a test image to verify that the region has been correctly coded
                region_folder = os.path.join(destination_folder, region.stem)
                region_json_path = os.path.join(region_folder, region.stem + ".json")
                image = ReadConverter.region_to_image(region)
                image_json = []
                for line_index, line in enumerate(image.lines):
                    LineAugmentation.normalize(line)
                    LineAugmentation.extend_backwards(line)
                    # LineAugmentation.extend(line, by=6, size_decay=0.9, confidence_decay=0.825)
                    LineAugmentation.enforce_minimum_height(line, minimum_height=32)
                    LineAugmentation.prevent_wrong_start(line, angle_threshold=30.0)
                    line_filename = os.path.join(region_folder, str(line.index) + ".png")
                    dewarped_line = Dewarper2(line.steps).dewarped(line.masked())
                    cv2.imwrite(line_filename, dewarped_line)
                    image_json.append({
                        "gt": line.text,
                        "image_path": line_filename,
                        "steps": [{
                            "upper_point": [step.upper_point.x, step.upper_point.y],
                            "base_point": [step.base_point.x, step.base_point.y],
                            "lower_point": [step.lower_point.x, step.lower_point.y],
                            "stop_confidence": step.stop_confidence,
                        } for step in line.steps],
                        "sol": {
                            "x0": line.steps[1].upper_point.x,
                            "x1": line.steps[1].lower_point.x,
                            "y0": line.steps[1].lower_point.y,
                            "y1": line.steps[1].lower_point.y,
                        }
                    })
                save_to_json(image_json, region_json_path)

        return None

    @staticmethod
    def parse_coords(xml, tag="Coords"):
        coords = xml.find(tag)["points"]
        coords = coords.split(" ")
        coords = [x.split(",") for x in coords]
        coords = [[int(a), int(b)] for [a, b] in coords]
        return coords

    @staticmethod
    def bbox(coords):
        min_x = min([x[1] for x in coords])
        max_x = max([x[1] for x in coords])
        min_y = min([x[0] for x in coords])
        max_y = max([x[0] for x in coords])

        return min_x, max_x, min_y, max_y

    @staticmethod
    def region_to_image(xml_region):
        image_data = {
            "index": xml_region.stem,
            "filename": xml_region.image_path,
            "lines": [line.steps for line in xml_region.lines if line.is_valid()]
        }

        return TrainingImage(image_data)


class XmlTextRegion:

    def __init__(self, index, img_path, xml):
        self.xml = xml
        self.stem = Path(img_path).stem + str(index)
        self.image_path = img_path
        self.coords = ReadConverter.parse_coords(xml)
        self.lines = [XmlTextLine(index, self, data) for index, data in enumerate(xml.findAll("TextLine")) if
                      data is not None]

    def save_image(self, path):
        image = ski.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        min_x, max_x, min_y, max_y = ReadConverter.bbox(self.coords)
        image = image[min_x:max_x, min_y:max_y]
        # img = img[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
        cv2.imwrite(path, image)

        try:
            painter = Painter(path=path)
        except Exception as e:
            print("Failed to save image", e)
            return

        for line in self.lines:
            if line.is_valid():
                line_step_data = to_steps(line)

                if line_step_data is not None:
                    painter.draw_line(
                        [p["upper_point"] for p in line_step_data["steps"]],
                        line_width=3,
                        color=(1, 0, 1, 1))
                    painter.draw_line(
                        [p["base_point"] for p in line_step_data["steps"]],
                        line_width=3,
                        color=(1, 0, 0, 1))
                    painter.draw_line(
                        [p["lower_point"] for p in line_step_data["steps"]],
                        line_width=3,
                        color=(1, 0, 1, 1))
            else:
                average = image.mean(axis=0).mean(axis=0)
                average = average / 255
                average = tuple(average)
                painter.draw_area([Point(coord[0], coord[1]) for coord in line.coords],
                                  fill_color=average)

            # if line.baseline is not None:
            #     painter.draw_line([Point(coord[0], coord[1]) for coord in line.baseline], line_width=3,
            #                       color=(1, 0, 0, 1))

        painter.save(path)
        # os.path.exists(path):
        #    os.remove(path)


class XmlTextLine:

    def __init__(self, index, region, xml):
        self.xml = xml
        self.index = index
        self.region = region
        min_x, max_x, min_y, max_y = ReadConverter.bbox(region.coords)
        self.coords = ReadConverter.parse_coords(xml)
        self.coords = [[y - min_y, x - min_x] for [y, x] in self.coords]
        self.baseline = ReadConverter.parse_coords(xml, "Baseline") if xml.find("Baseline") is not None else None
        self.baseline = [[y - min_y, x - min_x] for [y, x] in self.baseline] if self.baseline is not None else None
        self.text = xml.find("TextEquiv").find("Unicode").get_text() if xml.find("TextEquiv") is not None else None
        self.steps = to_steps(self)

    def is_valid(self):
        return self.steps is not None and len(self.steps) > 0


def slope(baseline):
    dy = baseline[1][1] - baseline[0][1]
    dx = baseline[1][0] - baseline[0][0]
    return dy / dx


def walk(start, slope, amount):
    new_x = start[0] + amount
    new_y = start[1] + amount * slope
    return [new_x, new_y]


def point_at(baseline, walked):
    return walk(baseline[0], slope(baseline), walked)


def to_steps(line):
    if line.baseline is None or line.coords is None:
        return None

    full_baseline = line.baseline
    hull = Polygon(line.coords)
    baseline_segment_count = len(full_baseline) - 1
    baseline_lenghts = [distance(full_baseline[i], full_baseline[i + 1]) for i in range(baseline_segment_count)]
    total_distance = sum(baseline_lenghts)
    line_data = {
        "index": line.index,
        "text": line.text,
        "steps": [],
    }

    walked = 0

    def get_point_from_walked(walked):
        acc = 0
        baseline_index = 0
        for index, length in enumerate(baseline_lenghts):
            acc += length
            if walked < acc:
                baseline_index = index
                acc -= length
                break
        baseline = full_baseline[baseline_index:baseline_index + 2]
        return baseline, point_at(baseline, walked - acc)

    while walked < total_distance:
        baseline, point = get_point_from_walked(walked)

        # Intersect hull to get upper and lower
        try:
            intersecting_line = perpendicular(point, baseline)
        except:
            return None

        if not hull.is_simple:
            print("[SELF INTERSECTING HULL]")
            return

        intersection = intersecting_line.intersection(hull)

        base_point = Point(point)
        upper_point = None
        lower_point = None

        if isinstance(intersection, MultiPoint) and len(intersection.bounds) == 4:
            upper_point = Point([intersection.bounds[0], intersection.bounds[1]])
            lower_point = Point([intersection.bounds[2], intersection.bounds[3]])
        elif isinstance(intersection, LineString) and len(intersection.bounds) == 4:
            upper_point = Point([intersection.bounds[0], intersection.bounds[1]])
            lower_point = Point([intersection.bounds[2], intersection.bounds[3]])
        elif isinstance(intersection, Point):
            print("Intersection was point, moving forward")
            walked += 4
            continue
        else:
            if walked < 40:
                walked += 4
            else:
                print("No intersection found ")
                return None

        if any([p is None for p in [base_point, lower_point, upper_point]]):
            continue

        height = base_point.distance(upper_point)

        height_threshold = 16

        angle = angle_between_points(base_point, upper_point)

        if height < height_threshold:
            # project upper point
            upper_point = get_new_point(base_point, angle, height_threshold)
            height = height_threshold

        # Check for lower point above base point
        lower_angle = angle_between_points(base_point, lower_point)

        if abs(angle - lower_angle) < 90:
            lower_point = base_point

        line_data["steps"].append({
            "upper_point": [upper_point.x, upper_point.y],
            "base_point": [base_point.x, base_point.y],
            "lower_point": [lower_point.x, lower_point.y],
        })
        walked += height

    return line_data
