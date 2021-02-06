from domain.running_line import RunningLine


class RunningImage:

    def __init__(self, path):
        self.path = path
        self.lines = []

    def add_sol(self, upper, lower):
        line = RunningLine(self)
        line.sol = [upper, lower]
        self.lines.append(line)
