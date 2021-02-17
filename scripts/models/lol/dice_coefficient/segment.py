import torch
from shapely.geometry import Point

from scripts.models.lol.dice_coefficient.vertex import Vertice


class LineSegment:

    def __init__(self, start, end):
        self.tensor = torch.stack([start, end])
        self.intersection_ids = []

    def add_intersection_id(self, id):
        self.intersection_ids.append(id)

    def starting_point(self):
        return Point(self.tensor[0][0].item(), self.tensor[0][1].item())

    def sort(self, map):
        self.intersection_ids.sort(key=lambda id: Point(map[id].tensor[0].item(), map[id][1].item()).distance(
            Point(self.tensor[0][0].item(), self.tensor[0][1].item())))

    def __getitem__(self, item):
        return self.tensor[item]

    def __len__(self):
        return len(self.tensor)

    def all(self, map):
        points = [Vertice(self.tensor[0])]
        points += [map[id] for id in self.intersection_ids]
        points.append(Vertice(self.tensor[1]))
        return points