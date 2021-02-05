# Overview

This project contains all the necessary scripts to compare the proposed method against the original version on which it
is inspired.

The `steps` folder contains bash scripts used for executing individual steps of the pipeline using the desired model and
dataset.

All bashed scripts demand two arguments:

|    |      |            | 
| ------------- |:-------------:| :-------------:| 
| -d     | --dataset     | iam, orcas or both |
| -m     | --model      | original, new or both      |  

# Pre-processing

### Requirements

The execution of the following pipeline requires the IAM and Orcalab data to be at `data/original/iam`
and `data/original/orcas`.

### Prepare images

Uses the ground truth data to extract lines from the datasets.

Generated lines can then be found at `data/[model]/[dataset]/pages`.

```bash
./steps/prepare_images.sh --dataset=both --model=both
```

### Generate character set

Creates character sets used for training the HTR models.

The character set file is stored at `data/[model]/[dataset]/pages/character_set.json`

```bash
./steps/generate_character_set.sh --dataset=both --model=both
```

# Training

The trained networks are stored at `scripts/[model]/snapshots/training/[network].pt`

### Train Start-of-Line finder

```bash
./steps/train_sol.sh --dataset=both --model=both
```

### Train Model (Line-Follower vs Line-Outliner)

```bash
./steps/train_model.sh --dataset=both --model=both
```

### Train HTR

```bash
./steps/train_htr.sh --dataset=both --model=both
```

# Aligned training

###### TODO

# Validation

###### TODO
