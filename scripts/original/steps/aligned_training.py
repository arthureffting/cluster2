import sys
import os
import multiprocessing

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
from subprocess import Popen

sys.path.append(os.getcwd())


def initial_alignment():
    Popen('python scripts/original/aligned/continuous_validation.py sample_config.yaml init')


def continuous():
    Popen('python scripts/original/aligned/continuous_validation.py sample_config.yaml')


def sol():
    Popen('python scripts/original/aligned/continuous_hw_training.py sample_config.yaml')


def hw():
    Popen('python scripts/original/aligned/continuous_hw_training.py sample_config.yaml')


def lf():
    Popen('python scripts/original/aligned/continuous_lf_training.py sample_config.yaml')


# First initial alignment synchronously
initial_alignment()
multiprocessing.Process(target=continuous).start()
multiprocessing.Process(target=sol).start()
multiprocessing.Process(target=hw).start()
multiprocessing.Process(target=lf).start()
