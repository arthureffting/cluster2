class SolParameters:

    def __init__(self):
        self.name = "sol-training"
        self.base0 = 16
        self.base1 = 16
        self.alpha_alignment = 0.1
        self.alpha_backprop = 0.1
        self.learning_rate = 0.0001
        self.crop_params = {
            "prob_label": 0.5,
            "crop_size": 256,
        }
        self.training_rescale_range = [384, 640]
        self.validation_rescale_range = [512, 512]  # Dont validate on random range
        self.batch_size = 1
        self.images_per_epoch = 1000
        self.stop_after_no_improvement = 1000
        self.dataset_split = 0.8
        self.max_epochs = 1000
