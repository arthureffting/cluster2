import sys
import os
import multiprocessing

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())


def initial_alignment():
    os.system('python scripts/original/aligned_fake/continuous_validation.py sample_config.yaml init')


def continuous():
    os.system('python scripts/original/aligned_fake/continuous_validation.py sample_config.yaml')


def sol():
    os.system('python scripts/original/aligned_fake/continuous_hw_training.py sample_config.yaml')


def hw():
    os.system('python scripts/original/aligned_fake/continuous_hw_training.py sample_config.yaml')


def lf():
    os.system('python scripts/original/aligned_fake/continuous_lf_training.py sample_config.yaml')


# First initial alignment synchronously
initial_alignment()
multiprocessing.Process(target=continuous).start()
multiprocessing.Process(target=sol).start()
multiprocessing.Process(target=hw).start()
multiprocessing.Process(target=lf).start()
