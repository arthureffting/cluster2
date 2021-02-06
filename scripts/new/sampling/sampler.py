from scripts.new.sampling.training_sample import TrainingSample


class FastSampler:

    def __init__(self, parameters):
        self.parameters = parameters
        self.samples = []
        self.interpolation = 0.0

    # Given a line, a point where to sample and a run length, return:
    # Current input
    # Desired output for run length
    def sample(self, line, until=1, run_length=1):

        steps_that_become_input = []  # These steps are used for getting the viewing windows
        backwards_runner = until
        backward_run_length = 0
        while backwards_runner >= 0 and backward_run_length < self.parameters.sequence_size:
            # Run backwards getting the viewing windows for the previous steps
            steps_that_become_input.append(line.steps[backwards_runner])
            backwards_runner -= 1
            backward_run_length += 1

        steps_that_become_target = []
        forwards_runner = until + 1
        forward_run_length = 0
        while forwards_runner < len(line.steps) and forward_run_length < run_length:
            steps_that_become_target.append(line.steps[forwards_runner])
            forwards_runner += 1
            forward_run_length += 1

        # The step beyond the targets is needed for calculating desired output
        step_beyond = line.steps[forwards_runner + 1] if forwards_runner < len(line.steps) - 1 else None

        steps_that_become_input.reverse()

        return TrainingSample(self.parameters,
                              line,
                              steps_that_become_input,
                              steps_that_become_target,
                              step_beyond=step_beyond)
