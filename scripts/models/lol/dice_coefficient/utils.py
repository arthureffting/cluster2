import torch
from shapely.geometry import Point

from scripts.utils.geometry import angle_between_points, get_new_point


def complete_polygons(sol, predicted_steps, desired_steps):
    predicted_polygon = torch.stack([sol[:4, :]] + [p[:] for p in predicted_steps])
    desired_polygon = torch.stack([sol[:4, :]] + [p[:4] for p in desired_steps])
    predicted_polygon, desired_polygon = steps_to_polygon(predicted_polygon), steps_to_polygon(desired_polygon)
    predicted_polygon, desired_polygon = desambiguate_polygons(predicted_polygon, desired_polygon)
    return predicted_polygon, desired_polygon


def steps_to_polygon(steps, indexes=[0, 2]):
    upper_points = torch.stack([p[indexes[0]] for p in steps])
    lower_points = torch.stack([p[indexes[1]] for p in steps])
    lower_points = torch.flip(lower_points, dims=[0])
    return torch.stack(
        [p for p in upper_points] + [p for p in lower_points])


def desambiguate_polygons(predicted_polygon, desired_polygon):
    """
    Since the Weiler-Atherton algorithm does not accept equal lines in both polygons,
    we have to shift the sol position a little to make it work
    """
    for index, predicted_point in enumerate(predicted_polygon):
        for desired_point in desired_polygon:
            if torch.all(torch.eq(predicted_point, desired_point)):
                # If something is similar, shift it so that the overlap algorithm works
                predicted_polygon[index] = torch.add(predicted_polygon[index], torch.tensor([-1, 1]).cuda())
    return predicted_polygon, desired_polygon


# Compare upper polygon
# Compare lower polygon


def shift_touching_upper_points(ground_truth):
    # Enforce a height of 2 to the ground truth lower polygon

    for index, step in enumerate(ground_truth):
        upper_as_point = Point(step[0][0].item(), step[0][1].item())
        base_as_point = Point(step[1][0].item(), step[1][1].item())
        lower_as_point = Point(step[2][0].item(), step[2][1].item())
        step_angle = angle_between_points(base_as_point, upper_as_point)
        lower_height = base_as_point.distance(lower_as_point)

        if lower_height < 2:
            lower_as_point = get_new_point(base_as_point, step_angle - 180, 2)

        # if index == 0:
        # If sol, push it backwards
        # angle_to_next = angle_between_points(base_as_point, Point(ground_truth[index + 1][1][0].item(),
        # ground_truth[index + 1][1][1].item()))
        # base_as_point = get_new_point(base_as_point, angle_to_next + 180, 2)
        # lower_as_point = get_new_point(lower_as_point, angle_to_next + 180, 2)

        # ground_truth[index][1][0] = torch.tensor(base_as_point.x).cuda()
        # ground_truth[index][1][1] = torch.tensor(base_as_point.y).cuda()
        ground_truth[index][2][0] = torch.tensor(lower_as_point.x).cuda()
        ground_truth[index][2][1] = torch.tensor(lower_as_point.y).cuda()

    return ground_truth


def lower_polygons(sol, predicted_steps, desired_steps):
    predicted_polygon = torch.stack([sol[:4, :]] + [p[:] for p in predicted_steps])
    desired_polygon_steps = torch.stack([sol[:4, :]] + [p[:4] for p in desired_steps])
    desired_polygon_steps = shift_touching_upper_points(desired_polygon_steps)
    predicted_polygon, desired_polygon = steps_to_polygon(predicted_polygon, indexes=[1, 2]), steps_to_polygon(
        desired_polygon_steps, indexes=[1, 2])
    predicted_polygon, desired_polygon = desambiguate_polygons(predicted_polygon, desired_polygon)
    return predicted_polygon, desired_polygon


def upper_polygons(sol, predicted_steps, desired_steps):
    # sol = shift_touching_upper_points([sol])[0]
    predicted_polygon = torch.stack([sol[:4, :]] + [p[:] for p in predicted_steps])
    desired_polygon = torch.stack([sol[:4, :]] + [p[:4] for p in desired_steps])
    predicted_polygon, desired_polygon = steps_to_polygon(predicted_polygon, indexes=[0, 1]), steps_to_polygon(
        desired_polygon, indexes=[0, 1])
    predicted_polygon, desired_polygon = desambiguate_polygons(predicted_polygon, desired_polygon)
    return predicted_polygon, desired_polygon
