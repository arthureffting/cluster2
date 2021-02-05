#!/bin/bash

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
    python3 "scripts/$MODEL_FOLDER/steps/$SCRIPT" --dataset="$DATASET_FOLDER"
  done
done
