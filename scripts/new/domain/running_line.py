from sampling.viewing_window import ViewingWindow
from utils.geometry import angle_between_points, get_new_point


class RunningLine:

    def __init__(self, image):
        self.steps = []
        self.image = image
        self.sol = None

    def starting_window(self, parameters):
        sol_angle = angle_between_points(self.sol[1], self.sol[0]) + 90
        sol_height = self.sol[0].distance(self.sol[1])
        backward_projected = get_new_point(self.sol[1], sol_angle - 180, sol_height)
        return ViewingWindow(parameters, self.image, backward_projected, sol_height, sol_angle)
