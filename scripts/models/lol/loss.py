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
    # angle_loss = torch.nn.MSELoss(reduction="sum")(predicted[:, 3], desired[:, 3].cuda())
    confidence_loss = torch.nn.MSELoss(reduction="sum")(predicted[:, 4], desired[:, 4].cuda())
    return (upper_loss + base_loss + lower_loss + confidence_loss) / predicted.shape[0]

