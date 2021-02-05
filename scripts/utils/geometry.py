import math
from shapely.geometry import Point


def angle_between_points(pt1, pt2):
    x_diff = pt2.x - pt1.x
    y_diff = pt2.y - pt1.y
    return math.degrees(math.atan2(y_diff, x_diff))


def get_new_point(pt, bearing, dist):
    bearing = math.radians(bearing)
    x = pt.x + dist * math.cos(bearing)
    y = pt.y + dist * math.sin(bearing)
    return Point(x, y)


def point_at_slope(base_point, slope, x_distance):
    return Point((base_point.coords[0] + x_distance), (base_point.coords[1] + (x_distance * slope)))


def midpoint(p1, p2):
    return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)


def midpoint_weighted(p1, p2, percentage):
    distance = p1.distance(p2)
    angle = angle_between_points(p1, p2)
    return get_new_point(p1, angle, distance * percentage)


def viewing_window_points(center, size, angle, ratio=1.0):
    ratio = ratio / 2
    upper = get_new_point(center, angle - 90, ratio * size)
    lower = get_new_point(center, angle + 90, ratio * size)
    upper_left = get_new_point(upper, angle - 180, ratio * size)
    upper_right = get_new_point(upper, angle, ratio * size)
    lower_left = get_new_point(lower, angle - 180, ratio * size)
    lower_right = get_new_point(lower, angle, ratio * size)

    return upper_left, upper_right, lower_right, lower_left

