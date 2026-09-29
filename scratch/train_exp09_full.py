import os
import sys
import torch
from ultralytics import YOLO
import shutil

def main():
    # Model YAML
    YAML = "models/yolo11-hrfusion.yaml"
    # Transferred E3 weights for EXP09 (as used in sanity)
    E3_WEIGHTS = "runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt"
    
    print("=" * 60)
    print("EXP09 YOLO11-L HR-Fusion FULL TRAINING (50 Epochs)")
    print("=" * 60)
    
    # 1. Build model
    print("\n[1] Building model from YAML...")
    model = YOLO(YAML)
    print("    Model built OK.")
    
    # 2. Load E3 pretrained weights
    print(f"\n[2] Loading E3 weights from {E3_WEIGHTS}...")
    try:
        model.load(E3_WEIGHTS)
        print("    Weight transfer OK.")
    except Exception as e:
        print(f"    Weight transfer FAILED: {e}")
        sys.exit(1)
        
    # 3. Train
    print("\n[3] Starting Training...")
    results = model.train(
        data="configs/visdrone.yaml",
        epochs=50,
        batch=4,
        imgsz=1536,
        project="runs/detect/experiments/model_search",
        name="EXP09_highres_fusion",
        seed=0,
        patience=30,
        optimizer="auto",
        cos_lr=False,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        erasing=0.4,
        auto_augment="randaugment",
        device="0"
    )
    
    print("\n[4] Training COMPLETE.")
    
if __name__ == "__main__":
    main()
