import subprocess
import sys
import os
import multiprocessing

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)

sys.path.append(os.getcwd())


def initial_alignment():
    subprocess.run([sys.executable,
                    'scripts/original/aligned/continuous_validation.py', 'sample_config.yaml', 'init'],
                   check=True)


def continuous():
    subprocess.run([sys.executable,
                    'scripts/original/aligned/continuous_validation.py', 'sample_config.yaml'],
                   check=True)


def sol():
    subprocess.run([sys.executable,
                    'scripts/original/aligned/continuous_sol_training.py', 'sample_config.yaml'],
                   check=True)


def hw():
    subprocess.run([sys.executable,
                    'scripts/original/aligned/continuous_hw_training.py', 'sample_config.yaml'],
                   check=True)


def lf():
    subprocess.run([sys.executable,
                    'scripts/original/aligned/continuous_lf_training.py', 'sample_config.yaml'],
                   check=True)


# First initial alignment synchronously
initial_alignment()
multiprocessing.Process(target=continuous).start()
multiprocessing.Process(target=sol).start()
multiprocessing.Process(target=hw).start()
multiprocessing.Process(target=lf).start()
