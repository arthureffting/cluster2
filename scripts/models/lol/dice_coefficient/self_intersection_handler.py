import torch
from shapely.geometry import Point

from scripts.models.lol.dice_coefficient.geometric import line_intersect


def ComplexPolygonHandler(clip):
    clip_segments = clip.lines()

    intersection_points = []
    intersection_map = []

    for i in range(len(clip_segments)):
        for j in range(len(clip_segments)):
            if i != j:
                intersection_point = line_intersect(clip_segments[i], clip_segments[j])
                if intersection_point is not None:
                    intersection_points.append(intersection_point)
                    intersection_map.append([i, j])

    loss = 0.0

    for i in range(len(intersection_points)):
        point = intersection_points[i]
        segment_indexes = intersection_map[i]
        first_segment, second_segment = clip_segments[segment_indexes[0]], clip_segments[segment_indexes[1]]
        tensors = [first_segment.tensor[0]] + [first_segment.tensor[1]] + [second_segment.tensor[0]] + [
            second_segment.tensor[1]]
        distance = [Point(point[0].item(), point[1].item()).distance(Point(p[0].item(), p[1].item())) for p in tensors]
        min_index = distance.index(min(distance))
        loss += torch.nn.MSELoss()(tensors[min_index], point)

    return torch.sub(torch.tensor(1).cuda(), torch.div(torch.tensor(1).cuda(), loss))