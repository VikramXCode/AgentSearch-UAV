#!/bin/bash
set -e

# Activate venv
source .venv/bin/activate

LOG_FILE="experiments/research/EXP09_highres_fusion/full_train.log"
VRAM_LOG="experiments/research/EXP09_highres_fusion/vram.log"

echo "Starting EXP09 full training (50 epochs) logging to $LOG_FILE"
echo "Monitoring VRAM to $VRAM_LOG"

# Start VRAM monitor
rm -f $VRAM_LOG
while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $VRAM_LOG; sleep 5; done &
VRAM_PID=$!

# Run python script
python scratch/train_exp09_full.py 2>&1 | tee $LOG_FILE

# Stop VRAM monitor
kill $VRAM_PID

# Compute peak VRAM
PEAK_VRAM=$(sort -nr $VRAM_LOG | head -1)
echo "Peak VRAM during training: ${PEAK_VRAM} MB" | tee -a $LOG_FILE
