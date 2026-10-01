import os
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Fix absolute path issues
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

def main():
    print("==================================================================")
    print("   🚀 MAXIMAL ACCURACY TRAINING: YOLO11X ON VISDRONE 2019       ")
    print("==================================================================")
    
    # 1. Base Model Selection
    # Using the absolute largest yolo11 series model for maximum parameter count
    # and feature extraction capability.
    model = YOLO("yolo11x.pt") 

    # 2. Aggressive Hyperparameters for Tiny Objects (VisDrone Specific)
    print("\nStarting Training with Extreme Hyperparameters...")
    
    results = model.train(
        # Data and Architecture
        data="configs/visdrone.yaml",
        project="runs/detect/experiments/maximal_accuracy",
        name="yolo11x_1536_extreme",
        epochs=150,
        patience=30,  # Plenty of patience for large models
        
        # Scaling
        imgsz=1536,  # Massive resolution to preserve tiny 10x10 pixel objects
        batch=4,     # Keep batch small to fit 1536px in 40GB A100 VRAM
        device=0,
        
        # Optimizer and Scheduler
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        cos_lr=True,
        workers=8,
        amp=True,
        
        # Advanced Augmentations for Dense/Tiny/Rare Objects
        mosaic=1.0,          # 100% mosaic to force learning contextual combinations
        mixup=0.15,          # Mixup helps with occlusion in dense crowds
        copy_paste=0.3,      # Crucial for rare classes (pastes trucks/tricycles into new scenes)
        degrees=10.0,
        translate=0.2,
        scale=0.6,
        shear=2.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        close_mosaic=20,     # Turn off mosaic for final 20 epochs to fine-tune on true distributions
        
        # Saving and Eval
        save_period=10,
        val=True,
        conf=0.001,          # Ensure high recall during internal mAP evaluation
        iou=0.6,
    )
    
    print("\nTraining completed successfully!")
    print(f"Results saved to: {results.save_dir}")

if __name__ == "__main__":
    main()
