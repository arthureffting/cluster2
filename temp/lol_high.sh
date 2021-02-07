#!/bin/bash
#SBATCH --job-name=sfrs-high
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH -o /cluster/%u/cluster2/slurm/results/%j.out
#SBATCH -e /cluster/%u/cluster2/slurm/results/%j.err
#Timelimit format: "hours:minutes:seconds"
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=effting@accuras.de

# Tell pipenv to install the virtualenvs in the cluster folder
export WORKON_HOME==/cluster/$(whoami)/.python_cache
export PYTHONPATH=.

# Small Python packages can be installed in own home directory. Not recommended for big packages like tensorflow -> Follow instructions for pipenv below
# cluster_requirements.txt is a text file listing the required pip packages (one package per line)

pip3 install --user -r requirements.txt

python3 scripts/new/training/train_model.py --dataset=iam --name=high_lr --images_per_epoch=5000 --testing_images_per_epoch=1000 --learning_rate=0.001
