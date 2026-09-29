#!/bin/bash
set -e

# Clean up sanity run
rm -rf runs/detect/runs/detect/experiments/model_search/EXP10_small_object

# Activate venv
source .venv/bin/activate

LOG_FILE="experiments/research/EXP10_small_object/full_train.log"
mkdir -p experiments/research/EXP10_small_object/

echo "Starting EXP10 FULL training (100 epochs) logging to $LOG_FILE"

# Run python script
python scratch/train_exp10_full.py 2>&1 | tee $LOG_FILE

echo "Training finished."
