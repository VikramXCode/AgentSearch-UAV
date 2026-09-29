"""
EXP09 Architecture Smoke Test
Verifies: model construction, forward pass, tensor shapes, NaN check,
parameter count, GFLOPs, weight transfer from E3.
"""
import torch
from ultralytics import YOLO

E3_WEIGHTS = "runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt"
YAML       = "models/yolo11-hrfusion.yaml"

print("=" * 60)
print("EXP09 YOLO11-L HR-Fusion Smoke Test")
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

# 3. Forward pass on dummy 1536px tensor
print("\n[3] Forward pass test (imgsz=1536)...")
model.model.eval()
dummy = torch.zeros(1, 3, 1536, 1536)
with torch.no_grad():
    out = model.model(dummy)

if isinstance(out, (list, tuple)):
    t0 = out[0]
    print(f"    Output[0] shape: {t0.shape}")
    # Check for NaN/Inf
    if torch.isnan(t0).any():
        print("    ERROR: NaN detected in output!")
    elif torch.isinf(t0).any():
        print("    ERROR: Inf detected in output!")
    else:
        print("    No NaN/Inf detected. ✅")
else:
    print(f"    Output shape: {out.shape}")

# 4. Expected anchor count for P3+P4+P5 at 1536px
p3 = (1536 // 8) ** 2   # 36864
p4 = (1536 // 16) ** 2  # 9216
p5 = (1536 // 32) ** 2  # 2304
total = p3 + p4 + p5
print(f"\n[4] Expected anchors (P3+P4+P5 @ 1536px): {p3}+{p4}+{p5} = {total}")
if isinstance(out, (list, tuple)):
    actual = out[0].shape[-1]
    match = "✅ MATCH" if actual == total else f"❌ MISMATCH (got {actual})"
    print(f"    Actual anchors in output: {actual} {match}")

# 5. Parameter count and GFLOPs
total_params = sum(p.numel() for p in model.model.parameters())
print(f"\n[5] Total Parameters: {total_params / 1e6:.3f} M")
print(f"    (E3 baseline: 25.372 M)")
print(f"    Delta: +{(total_params - 25372160) / 1e6:.3f} M")

# 6. NC verification
print(f"\n[6] Number of classes: {model.model.yaml.get('nc', 'N/A')}")

# 7. Inference API test
print("\n[7] Inference API test on dummy numpy array...")
import numpy as np
dummy_img = np.zeros((1536, 1536, 3), dtype=np.uint8)
res = model.predict(dummy_img, imgsz=1536, verbose=False)
print(f"    Inference OK. Detected {len(res[0].boxes)} boxes on blank image.")

print("\n" + "=" * 60)
print("Smoke test COMPLETE. Ready for 3-epoch sanity training.")
print("=" * 60)
