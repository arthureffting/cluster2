##
## The
##
import torch


def weighted(predicted, desired, percentage):
    upper_loss = torch.nn.MSELoss(reduction="sum")(predicted[:, 0], desired[:, 0].cuda())
    base_loss = torch.nn.MSELoss(reduction="sum")(predicted[:, 1], desired[:, 1].cuda())
    lower_loss = torch.nn.MSELoss(reduction="sum")(predicted[:, 2], desired[:, 2].cuda())
    # angle_loss = torch.nn.MSELoss(reduction="sum")(predicted[:, 3], desired[:, 3].cuda())
    confidence_loss = torch.nn.MSELoss(reduction="sum")(predicted[:, 4], desired[:, 4].cuda()) * percentage
    return (upper_loss + base_loss + lower_loss + confidence_loss) / predicted.shape[0]


def standard(predicted, desired):
    assert predicted.shape[0] == desired.shape[0]
    upper_loss = torch.nn.MSELoss(reduction="sum")(predicted[:, 0], desired[:, 0].cuda())
    base_loss = torch.nn.MSELoss(reduction="sum")(predicted[:, 1], desired[:, 1].cuda())
    lower_loss = torch.nn.MSELoss(reduction="sum")(predicted[:, 2], desired[:, 2].cuda())
    angle_loss = torch.nn.MSELoss(reduction="sum")(predicted[:, 3], desired[:, 3].cuda())
    confidence_loss = torch.nn.MSELoss(reduction="sum")(predicted[:, 4], desired[:, 4].cuda())
    return (upper_loss + base_loss + lower_loss + angle_loss + confidence_loss) / predicted.shape[0]


def outline_loss(predicted, desired):
    return torch.nn.MSELoss()(predicted[:, [0, 2]], desired[:, [0, 2]].cuda())


def stop_loss(predicted, desired):
    return torch.nn.MSELoss()(predicted[:, 4, 0], desired[:, 4, 0].cuda())


def prediction_to_polygon_coords(sol, prediction):
    upper_points = torch.stack([sol[0]] + [p[0] for p in prediction])
    lower_points = torch.stack([sol[2]] + [p[2] for p in prediction])
    lower_points = torch.flip(lower_points, [0])
    polygon_points = torch.stack([p for p in upper_points] + [p for p in lower_points])
    return polygon_points


# Weights each SME using the difference in confidence prediction
def distributed_weights(predicted, desired):
    assert predicted.shape[0] == desired.shape[0]
    desired = desired.cuda()
    confidence_diffs = torch.add(torch.abs(torch.sub(predicted[:, 4, 0], desired[:, 4, 0])), torch.tensor(1).cuda())
    loss_fn = torch.nn.MSELoss()
    loss = 0.0
    for index in range(predicted.shape[0]):
        loss += torch.mul(loss_fn(predicted[index, [0, 2]], desired[index, [0, 2]]), confidence_diffs[index])
    return loss


def separate_losses(predicted, desired):
    return outline_loss(predicted, desired) + stop_loss(predicted, desired)
