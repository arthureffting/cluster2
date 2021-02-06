
class SamplingParameters():

    def __init__(self, name):
        super(SamplingParameters, self).__init__()
        self.name = name
        self.sequence_size = 6  # The number of viewing windows used in the ConvLSTM
        self.training_run_length = 1  # Maximum number of steps to take while training
        self.stop_threshold = 0.85  # Stop confidence threshold where runners stop running
        self.patch_ratio = 5  # How much the extracted path is bigger than the step's size
        self.patch_size = 64  # Size of the sides of the patch in pixels, after resizing
        self.min_step_size = 16  # Minimum step size allowed, this prevents the network from converging to 0
        self.origin_distortion = 0.2  # Amount of pixels that the viewing window focus can be shifted while training
        self.angle_distortion = 30  # Amount of degrees that the viewing window can be tilted during training
        self.size_distortion = 0.1  # Percentage of the size that can be decreased or increased during training
        self.validation_steps = 1000  # Steps used in validation when validating against augmented steps
        self.hw_height = 64  # Height of the images fed to the HTR module
        self.learning_rate = 0.0001
        self.dataset_split = 0.80
        self.epoch_size = 100
        self.stop_after_no_improvement = 100



    @staticmethod
    def DefaultLOL(name: "default-lol"):
        parameters = SamplingParameters(name=name)
        parameters.stop_after_no_improvement = 50
        parameters.steps_per_epoch = 1000
        parameters.validation_steps = 200
        parameters.learning_rate = 0.00001
        parameters.sequence_size = 5
        parameters.patch_size = 32
        parameters.batch_size = 1
        parameters.min_step_size = 32
        parameters.origin_distortion = 0.1  # Amount of pixels that the viewing window focus can be shifted while training
        parameters.angle_distortion = 12.5  # Amount of degrees that the viewing window can be tilted during training
        parameters.size_distortion = 0.1
        return parameters
