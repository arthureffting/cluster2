class RunningStep:

    def __init__(self, upper, base, lower, stop, angle):
        self.upper_point = upper
        self.base_point = base
        self.lower_point = lower
        self.stop_confidence = stop
        self.angle = angle

    def calculate_upper_height(self):
        return self.base_point.distance(self.upper_point)

    def calculate_lower_height(self):
        return self.base_point.distance(self.lower_point)
