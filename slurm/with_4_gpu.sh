#!/bin/bash
#SBATCH --job-name=sfrs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --mem=12000
#SBATCH -o /cluster/%u/sfrs/slurm/results/%j.out
#SBATCH -e /cluster/%u/sfrs/slurm/results/%j.err
#Timelimit format: "hours:minutes:seconds"
#SBATCH --time=02:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=effting@accuras.de





# Tell pipenv to install the virtualenvs in the cluster folder

export WORKON_HOME==/cluster/$(whoami)/.python_cache
export PYTHONPATH=.

# Small Python packages can be installed in own home directory. Not recommended for big packages like tensorflow -> Follow instructions for pipenv below
# cluster_requirements.txt is a text file listing the required pip packages (one package per line)

pip3 install --user -r requirements.txt

sh ${@}
