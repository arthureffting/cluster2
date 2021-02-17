import torch

from scripts.models.lol.dice_coefficient.dice_polygon import DicePolygon
from scripts.models.lol.dice_coefficient.self_intersection_handler import ComplexPolygonHandler
from scripts.models.lol.dice_coefficient.weiler_atherton import WeilerAtherton


def DiceCoefficientLoss(predicted, desired, overlap="weiler-atherton"):
    """

    Calculates the inverse dice coefficient between a predicted and desired polygon shapes.
    This algorithm assumes the desired shape to be simple (not self-intersecting).
    Self-intersecting clips will be subject to a different loss function, which roughly calculates
    how much vertices would have to be translated to avoid each intersection

    :param predicted: Ordered polygon coordinates in the shape ( N, 2 ) where N is the number of vertices in the clip polygon
    :param desired: Ordered polygon coordinates in the shape ( N, 2 ) where N is the number of vertices in the clipped polygon
    :param overlap: Algorithm used for calculating the overlap between the polygons
    :return: Tuple (loss, overlap) containing the inverse Sørensen–Dice coefficient for the two given polygons and a list of the overlapping areas

    """

    if overlap.lower() != "weiler-atherton":
        raise Exception("This loss function currently only supports the Weiler-Atherton algorithm.")

    clip = DicePolygon(predicted)
    subj = DicePolygon(desired)

    if not subj.is_simple():
        raise Exception("The desired polygon shape can not be self-intersecting.")

    if not clip.is_simple():
        # Clip is not simple, punish intersections
        return ComplexPolygonHandler(clip), []

    total_area = torch.add(clip.area(), subj.area())
    overlap_area = torch.tensor(0).cuda()
    intersections = WeilerAtherton(clip, subj)

    if intersections is not None:
        overlapping_areas = torch.stack([i.area() for i in intersections])
        overlap_area = torch.sum(overlapping_areas)
    else:
        intersections = []

    intersections = [i for i in intersections if len(i) > 3]

    dice_coefficient_value = torch.div(torch.mul(overlap_area, 2), total_area)
    inverse_value = torch.sub(torch.tensor(1).cuda(), dice_coefficient_value)

    return inverse_value, intersections
