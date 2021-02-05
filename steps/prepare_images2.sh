#!/bin/bash
#SBATCH --job-name=PREPARE_IMAGES_EFFTING
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH -o /cluster/%u/sfrs/slurm/results/%j.out
#SBATCH -e /cluster/%u/sfrs/slurm/results/%j.err
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds"
#SBATCH --time=02:30:00

# Tell pipenv to install the virtualenvs in the cluster folder

export WORKON_HOME==/cluster/$(whoami)/.python_cache
export PYTHONPATH=.

# Small Python packages can be installed in own home directory. Not recommended for big packages like tensorflow -> Follow instructions for pipenv below
# cluster_requirements.txt is a text file listing the required pip packages (one package per line)

pip3 install --user -r requirements.txt

export DATA_FOLDER="data"
export ORIGINAL_FOLDER="${DATA_FOLDER}/original"
export SPLIT_FOLDER="${DATA_FOLDER}/split"

IAM="iam"
ORCAS="orcas"
START_FOLLOW_READ="original"
SFR_STOP="new"
BOTH="both"

# Parse arguments
for i in "$@"; do
  case $i in
  -d=* | --dataset=*)
    DATASET="${i#*=}"
    shift
    ;;
  -m=* | --model=*)
    MODEL="${i#*=}"
    shift
    ;;
  -s=* | --script=*)
    SCRIPT="${i#*=}"
    shift
    ;;
  *) ;;

  esac
done

if ! [[ "$DATASET" =~ ^(${IAM}|${ORCAS}|${BOTH})$ ]]; then
  echo "Invalid dataset '${DATASET}'. Choose between '${IAM}', '${ORCAS}' or '${BOTH}'"
  exit 1
fi

if ! [[ "$MODEL" =~ ^(${START_FOLLOW_READ}|${SFR_STOP}|${BOTH})$ ]]; then
  echo "Invalid model '${MODEL}'. Choose between '${START_FOLLOW_READ}', '${SFR_STOP}' or '${BOTH}'"
  exit 1
fi

MODEL_FOLDERS=()
DATASET_FOLDERS=()

if [[ $MODEL == "$BOTH" ]]; then MODEL_FOLDERS+=("${START_FOLLOW_READ}" "${SFR_STOP}"); else MODEL_FOLDERS+=("${MODEL}"); fi
if [[ $DATASET == "$BOTH" ]]; then DATASET_FOLDERS+=("$IAM" "$ORCAS"); else DATASET_FOLDERS+=("${DATASET}"); fi

for MODEL_FOLDER in "${MODEL_FOLDERS[@]}"; do
  for DATASET_FOLDER in "${DATASET_FOLDERS[@]}"; do
    python "scripts/$MODEL_FOLDER/steps/prepare_images.py" --dataset="$DATASET_FOLDER"
  done
done
