from shapely.validation import explain_validity
from shapely.geometry import Polygon, LineString, Point
import numpy as np

from scripts.utils.geometry import angle_between_points, get_new_point


class GtLine:

    def __init__(self, polygon, baseline, text, height_threshold=0.8):
        self.polygon = GtPolygon(self, polygon)
        self.baseline = GtBaseline(self, baseline, height_threshold=height_threshold)
        self.text = text

    def steps(self):
        return map(lambda x: x.format(), self.baseline.intersections())


class GtPolygon:

    def __init__(self, line, polygon_data):
        self.polygon = Polygon(polygon_data)
        explain_validity(self.polygon)
        self.line = line

    def get(self):
        return self.polygon


class GtBaseline:

    def __init__(self, line, baseline, height_threshold):
        self.line = line
        self.baseline = baseline
        self.segments = []

        for i in range(0, len(baseline) - 1):
            p1 = baseline[i]
            p2 = baseline[i + 1]
            self.segments.append(GtBaselineSegment(self, p1, p2, height_threshold))

    def intersections(self):

        # For each segment of the baseline, get its intersections
        # and pass the rest to the next segment
        segments_intersections = []
        rest = 0
        relevant_segments = list(filter(lambda x: x.segment.intersects(self.line.polygon.get()), self.segments))

        # Trim leading segments
        while len(relevant_segments) > 0 and relevant_segments[0].get_first_section() is None:
            relevant_segments = relevant_segments[1:]

        for i in range(0, len(relevant_segments)):
            segment = relevant_segments[i]
            try:
                this_segment_intersections, rest = segment.intersections(offset=rest, start_of_line=(i == 0))
                segments_intersections += this_segment_intersections
            except NoStartOfLine as e:
                print("[Segment #" + str(i) + "] [Does not have a start of line]")

        relevant_segments.reverse()

        for i, segment in enumerate(relevant_segments):
            last_section = segment.get_last_section()
            if last_section is not None:
                segments_intersections = list(
                    filter(lambda x: x.segment != segment or x.distance < last_section.distance,
                           segments_intersections))
                # segments_intersections.append(last_section)
                break

        segment_count = len(segments_intersections)

        if segment_count >= 1:
            last_section = segments_intersections[-1]
            reference = last_section
            last_angle = (90.0 + angle_between_points(reference.p1, reference.p2))
            distance_multipler = 2 / 3
            p1 = get_new_point(reference.p1, last_angle, distance_multipler * reference.height())
            p2 = get_new_point(reference.p2, last_angle, distance_multipler * reference.height())
            new_section = VirtualSegmentSection(p1, p2, reference.confidence)
            segments_intersections.append(new_section)
            segments_intersections[-1].confidence = 0.0

        return segments_intersections


class NoStartOfLine(Exception):
    pass


class GtBaselineSegment:
    # The trimming threshold defines the step size
    # for the search of first and last vertical sections
    TRIMMING_THRESHOLD_STEP_SIZE = 0.1

    # The walking step size defines the rate with which the algorithm
    # moves forward when creating the ground truth
    WALKING_STEP_SIZE = 4
    WALKING_HEIGHT_MULTIPLIER_THRESHOLD = 0.5

    def __init__(self, baseline, p1, p2, height_threshold):
        self.height_threshold = height_threshold
        self.baseline = baseline
        self.start = p1
        self.start_point = Point(*p1)
        self.end = p2
        self.end_point = Point(*p2)
        self.length = Point(*self.start).distance(Point(*self.end))
        self.segment = LineString([p1, p2])
        self.slope = (self.end[1] - self.start[1]) / (self.end[0] - self.start[0])
        self.bearing = angle_between_points(Point(*p1), Point(*p2))
        self.vertical_sections = []

    def get_average_height(self, step_size=5):
        steps = np.arange(0, self.length, step_size)
        height_sum = 0
        height_count = 0
        for step in steps:
            section = VerticalSegmentSection(self, distance=step)
            if section.intersects():
                height_sum += section.height()
                height_count += 1
        return 0 if height_count == 0 else height_sum / height_count

    def get_first_section(self, minimum_height=None):

        if minimum_height is None:
            minimum_height = self.height_threshold * self.get_average_height()

        steps = np.arange(0, self.length, self.TRIMMING_THRESHOLD_STEP_SIZE)
        for step in steps:
            section = VerticalSegmentSection(self, distance=step)
            if section.intersects() and section.height() > minimum_height:
                return section
        return None

    def get_last_section(self, minimum_height=None):

        if minimum_height is None:
            minimum_height = self.height_threshold * self.get_average_height()

        steps = np.arange(self.length, 0, -self.TRIMMING_THRESHOLD_STEP_SIZE)
        for step in steps:
            section = VerticalSegmentSection(self, distance=step)
            if section.intersects() and section.height() > minimum_height:
                section.set_confidence(0.1)
                return section
        return None

    def intersections(self, offset=0, start_of_line=False):

        if not self.segment.intersects(self.baseline.line.polygon.get()):
            # print("Non intersecting segment")
            return [], 0.0

        if start_of_line:
            reference = self.get_first_section()
            if reference is None:
                raise NoStartOfLine("Start of line does not intersect polygon at any point.")
        else:
            reference = VerticalSegmentSection(self, distance=offset)

        if not reference.intersects():
            # print("Non intersecting segment")
            return [], 0.0

        intersections = [reference]
        walking = True
        maximum_relative_step = reference.height()
        relative_step = 0
        exceed_amount = 0

        while walking:

            # Adds a one step size to the current distance
            relative_step += self.WALKING_STEP_SIZE
            total_step = reference.distance + relative_step

            # sys.stdout.write("\r" + "[Segment walk] " + str(round(100 * total_step / self.length, 2)) + "%")

            # The new step exceeds the height of the reference
            if relative_step >= maximum_relative_step:
                new_reference_distance = reference.distance + maximum_relative_step
                if new_reference_distance <= self.length:
                    # Set new reference
                    reference = VerticalSegmentSection(self, distance=reference.distance + maximum_relative_step)
                    if reference.intersects():
                        intersections.append(reference)
                        maximum_relative_step = reference.height()
                        relative_step = 0
                        continue
                else:
                    exceed_amount = new_reference_distance - self.length
                    walking = False
                    break

            # This step will exceed the total length, stop here
            if total_step > self.length:
                exceed_amount = (reference.distance + maximum_relative_step) - self.length
                walking = False
                break

            # Gets the new section
            section = VerticalSegmentSection(self, distance=total_step)
            if section.intersects():
                height_difference = abs(section.height() - reference.height())
                if height_difference > self.WALKING_HEIGHT_MULTIPLIER_THRESHOLD * reference.height():
                    reference = section
                    intersections.append(reference)
                    maximum_relative_step = reference.height()
                    relative_step = 0
                    continue

        return intersections, exceed_amount

    def interpolate(self, distance):
        return self.segment.interpolate(distance, normalized=False)


class FormatablePoint:

    def format(self):
        pass

    def height(self):
        pass


class VirtualSegmentSection(FormatablePoint):

    def __init__(self, p1, p2, confidence):
        self.p1 = p1
        self.p2 = p2
        self.confidence = confidence

    def height(self):
        return self.p1.distance(Point(self.p2))

    def format(self):
        return {
            "x0": self.p1.x,
            "y0": self.p1.y,
            "x1": self.p2.x,
            "y1": self.p2.y,
            "confidence": self.confidence,
        }


class VerticalSegmentSection(FormatablePoint):
    MAXIMUM_SCAN_HEIGHT = 1000

    def __init__(self, segment, distance=0):
        self.distance = distance
        self.segment = segment
        self.confidence = 1.0
        interpolation_distance = min(self.distance, self.segment.length)
        self.point = self.segment.interpolate(interpolation_distance)
        scan_p1 = get_new_point(self.point, self.segment.bearing + 90, self.MAXIMUM_SCAN_HEIGHT)
        scan_p2 = get_new_point(self.point, self.segment.bearing - 90, self.MAXIMUM_SCAN_HEIGHT)
        scan_line = LineString([scan_p1, scan_p2])
        try:
            self.intersection_line = scan_line.intersection(self.segment.baseline.line.polygon.get())
            self.p1 = Point(self.intersection_line.coords[0][0], self.intersection_line.coords[0][1])
            self.p2 = Point(self.intersection_line.coords[1][0], self.intersection_line.coords[1][1])
        except Exception as e:
            self.intersection_line = None

    def set_confidence(self, confidence):
        self.confidence = confidence

    def height(self):
        return self.p1.distance(Point(self.p2))

    def intersection(self):
        return self.intersection_line

    def intersects(self):
        return isinstance(self.intersection_line, LineString) and len(self.intersection_line.coords) == 2

    def format(self):
        return {
            "x0": self.intersection_line.coords[0][0],
            "y0": self.intersection_line.coords[0][1],
            "x1": self.intersection_line.coords[1][0],
            "y1": self.intersection_line.coords[1][1],
            "confidence": self.confidence,
        }
