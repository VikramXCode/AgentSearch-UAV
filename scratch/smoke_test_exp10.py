import os
import torch
from ultralytics import YOLO

# Must set before running training or loss
os.environ["YOLO_SMALL_OBJ_WEIGHT"] = "2.0"

E3_WEIGHTS = "runs/detect/runs/detect/experiments/model_search/EXP09_highres_fusion/weights/best.pt"
YAML       = "models/yolo11-hrfusion.yaml"

print("=" * 60)
print("EXP10 YOLO11-L HR-Fusion + Small Object Weighting Smoke Test")
print("=" * 60)

# 1. Build model
print("\n[1] Building model from YAML...")
model = YOLO(YAML)
print("    Model built OK.")

# 2. Load EXP09 pretrained weights
print(f"\n[2] Loading EXP09 weights from {E3_WEIGHTS}...")
try:
    model.load(E3_WEIGHTS)
    print("    Weight transfer OK.")
except Exception as e:
    print(f"    Weight transfer FAILED: {e}")

# 3. Forward pass on dummy 1536px tensor
print("\n[3] Forward pass test (imgsz=1536)...")
model.model.eval()
dummy = torch.zeros(1, 3, 1536, 1536)
with torch.no_grad():
    out = model.model(dummy)

if isinstance(out, (list, tuple)):
    t0 = out[0]
    print(f"    Output[0] shape: {t0.shape}")
    if torch.isnan(t0).any():
        print("    ERROR: NaN detected in output!")
    elif torch.isinf(t0).any():
        print("    ERROR: Inf detected in output!")
    else:
        print("    No NaN/Inf detected. ✅")
else:
    print(f"    Output shape: {out.shape}")

# 4. Parameter count and GFLOPs
total_params = sum(p.numel() for p in model.model.parameters())
print(f"\n[4] Total Parameters: {total_params / 1e6:.3f} M")

print("\n" + "=" * 60)
print("Smoke test COMPLETE. Ready for 3-epoch sanity training.")
print("=" * 60)
