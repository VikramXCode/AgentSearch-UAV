#!/bin/bash
set -e

# Activate venv
source .venv/bin/activate

LOG_FILE="experiments/research/EXP10_small_object/sanity_train.log"
mkdir -p experiments/research/EXP10_small_object/

echo "Starting EXP10 sanity training (3 epochs) logging to $LOG_FILE"

# Run python script
python scratch/train_exp10_sanity.py 2>&1 | tee $LOG_FILE

echo "Training finished."
