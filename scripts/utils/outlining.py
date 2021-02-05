import torch


def scale(vector, ratio):
    torch.cuda.empty_cache()
    scaling_matrix = scaling_tensor(ratio)
    resulting_vector = torch.mm(scaling_matrix, vector.cuda())
    return resulting_vector.view(2, 1)


# Returns a rotated vector
def rotate(vector, angle):
    rotation_matrix = rotation_tensor(angle)
    resulting_vector = torch.mm(rotation_matrix, vector.cuda())
    return resulting_vector.view(2, 1)


# Creates a two dimensional scaling matrix with the given a ratio
def scaling_tensor(ratio):
    as_tensor = torch.tensor([ratio], dtype=torch.float32, device=0)
    return torch.stack([
        torch.stack([as_tensor, torch.zeros(1, device=0)]),
        torch.stack([torch.zeros(1, device=0), as_tensor])]).view(2, 2)


# Creates a two dimensional rotation matrix with the given angle in degrees
def rotation_tensor(angle):
    angle = torch.deg2rad(-angle)
    cos = torch.tensor([torch.cos(angle)], dtype=torch.float32, device=0)
    sin = torch.tensor([torch.sin(angle)], dtype=torch.float32, device=0)
    return torch.stack([
        torch.stack([cos, -sin]),
        torch.stack([sin, cos])]).reshape(2, 2)


def tensor_point(x, y):
    return torch.tensor([[x], [y]], dtype=torch.float32, device=0)