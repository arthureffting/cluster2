import torch
from shapely.geometry import Polygon

from scripts.models.lol.dice_coefficient.segment import LineSegment


class DicePolygon:

    def __init__(self, coordinates):
        self.tensor = coordinates

    def lines(self):
        lines = []
        for i in range(len(self.tensor)):
            last = i == len(self.tensor) - 1
            lines.append(LineSegment(self.tensor[i], self.tensor[0 if last else i + 1]))
        return lines

    def area(self):
        """
        Implementation of the shoe-lace algorithm for calculating a polygon's area.
        :return: Total area of the current polygon
        """
        return 0.5 * torch.abs(
            torch.dot(self.x(), torch.roll(self.y(), 1)) - torch.dot(self.y(), torch.roll(self.x(), 1)))

    def x(self):
        return torch.stack([t[0] for t in self.tensor])

    def y(self):
        return torch.stack([t[1] for t in self.tensor])

    def as_shapely(self):
        return Polygon(self.tensor)

    def __getitem__(self, item):
        return self.tensor[item]

    def __len__(self):
        return len(self.tensor)

    def is_simple(self):
        return self.as_shapely().is_simple

    def clone(self):
        return self.tensor.clone()
