import torch


class Vertice:

    def __init__(self, point):
        self.tensor = point
        self.is_intersection = False

    def __getitem__(self, item):
        return self.tensor[item]

    def __len__(self):
        return len(self.tensor)

    def same_as(self, other_vertex):
        result = torch.all(torch.eq(self.tensor, other_vertex.tensor))
        return result.item()

    def to_string(self):
        return "[" + str(self.tensor[0].item()) + ", " + str(self.tensor[1].item()) + "]"