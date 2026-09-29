#!/usr/bin/env python3
"""
Fine-tune YOLO11l on VisDrone to push mAP@50 >= 0.50.
Run: python train_yolo11l_visdrone.py
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# ── imports ──────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ERROR: ultralytics not installed. Run: pip install ultralytics")

# ── config ───────────────────────────────────────────────────────────────────
MODEL_WEIGHTS = "yolo11l.pt"           # base pretrained model (already in project root)
DATA_YAML     = "configs/visdrone_yolo_prepared.yaml"
TRAIN_CONFIG  = "configs/train_yolo11l_visdrone.yaml"

# Verify files exist
for f in [MODEL_WEIGHTS, DATA_YAML]:
    if not Path(f).exists():
        sys.exit(f"ERROR: required file not found: {f}")

print("=" * 60)
print(" AgentSearch-UAV — YOLO11l VisDrone Fine-Tune")
print("=" * 60)
print(f"  Base weights : {MODEL_WEIGHTS}")
print(f"  Dataset      : {DATA_YAML}")
print(f"  Config       : {TRAIN_CONFIG}")
print()

# ── GPU check ────────────────────────────────────────────────────────────────
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU          : {gpu} ({mem:.1f} GB)")
else:
    print("  WARNING: No GPU detected — training will be very slow on CPU")
print()

# ── load model ───────────────────────────────────────────────────────────────
model = YOLO(MODEL_WEIGHTS)

# ── train ─────────────────────────────────────────────────────────────────────
t0 = time.time()

results = model.train(
    data=DATA_YAML,
    project="runs/detect",
    name="yolo11l_visdrone_finetune",
    epochs=100,
    imgsz=1280,
    batch=8,
    device=0,
    workers=8,
    seed=42,
    # convergence
    patience=20,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=3,
    warmup_momentum=0.8,
    # regularisation
    weight_decay=0.0005,
    label_smoothing=0.1,
    # augmentation (aerial-optimised)
    degrees=10.0,
    translate=0.1,
    scale=0.9,
    shear=2.0,
    perspective=0.0005,
    flipud=0.1,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.3,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    close_mosaic=15,
    # NMS / conf (keep low for mAP eval)
    conf=0.001,
    iou=0.65,
    # misc
    amp=True,
    plots=True,
    exist_ok=True,
    verbose=True,
    save_period=10,
    rect=False,
)

elapsed = time.time() - t0
print()
print("=" * 60)
print(f" Training complete in {elapsed/3600:.1f} hours")
print("=" * 60)

# ── validate best weights ─────────────────────────────────────────────────────
best_weights = Path("runs/detect/yolo11l_visdrone_finetune/weights/best.pt")
if best_weights.exists():
    print(f"\n Validating best weights: {best_weights}")
    best_model = YOLO(str(best_weights))
    val_results = best_model.val(
        data=DATA_YAML,
        imgsz=1280,
        batch=8,
        device=0,
        conf=0.001,
        iou=0.65,
        split="val",
        plots=True,
        save_json=True,
    )
    map50    = val_results.box.map50
    map50_95 = val_results.box.map
    print()
    print("=" * 60)
    print(f"  FINAL mAP@50    : {map50:.4f}  ({map50*100:.1f}%)")
    print(f"  FINAL mAP@50-95 : {map50_95:.4f}  ({map50_95*100:.1f}%)")
    print(f"  Best weights    : {best_weights}")
    print("=" * 60)

    # ── copy best weights to project weights/ dir ──────────────────────────
    import shutil
    dest = Path("weights/yolo11l_visdrone_best.pt")
    shutil.copy(best_weights, dest)
    print(f"\n  Best weights saved → {dest}")
else:
    print("  WARNING: best.pt not found, check training logs")
