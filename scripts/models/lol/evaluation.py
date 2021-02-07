# Both tensors in the form
# [ [ upper_point, base_point, lower_point, angle, stop_confidence ] ]
from shapely.geometry import Polygon


def outline(predictions):
    upper_points = [(p[0][0].item(), p[0][1].item()) for p in predictions]
    lower_points = [(p[2][0].item(), p[2][1].item()) for p in predictions]
    lower_points.reverse()
    points = upper_points + lower_points
    points += [upper_points[0]]
    return points


def to_polygon(predictions):
    return Polygon(outline(predictions))


def evaluation_cost(predicted, desired):
    try:
        prediction_polygon = to_polygon(predicted)
        desired_polygon = to_polygon(desired)
        intersection = desired_polygon.intersection(prediction_polygon)
        return (2 * intersection.area) / (prediction_polygon.area + desired_polygon.area)
    except:
        return 0
