#!/bin/bash
# Launch training in tmux
tmux new-session -s exp08_p2_sanity -d '.venv/bin/yolo train model=models/yolo11-p2.yaml pretrained=runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt data=configs/visdrone.yaml epochs=3 imgsz=1024 batch=4 project=runs/detect/experiments/model_search name=EXP08_yolo11l_p2_1024_sanity | tee experiments/research/EXP08_p2_1024_sanity.log'

# Monitor VRAM and wait for tmux session to end
MAX_VRAM=0
echo "Monitoring training..."
while tmux has-session -t exp08_p2_sanity 2>/dev/null; do
    # Get memory usage of the python process
    current_vram=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | sort -n | tail -1)
    if [ ! -z "$current_vram" ]; then
        if [ "$current_vram" -gt "$MAX_VRAM" ]; then
            MAX_VRAM=$current_vram
        fi
    fi
    sleep 30
done

echo "Training finished."
echo "Max VRAM used: ${MAX_VRAM} MiB" > experiments/research/EXP08_p2_1024_sanity_vram.txt
