from scripts.models.lol.dice_coefficient.vertex import Vertice


class Intersection(Vertice):

    def __init__(self, id, point):
        super(Intersection, self).__init__(point)
        self.is_intersection = True
        self.id = id
        self.entering = False
        self.leaving = not self.entering
