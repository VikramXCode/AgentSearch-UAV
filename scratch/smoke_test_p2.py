import torch
from ultralytics import YOLO

# 1. Verify model construction
print("Building YOLO11-P2 model...")
model = YOLO("models/yolo11-p2.yaml")

# Load pretrained weights from E3 (ignoring mismatches for the new head)
e3_weights = "runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt"
print(f"Loading E3 weights from {e3_weights}...")
try:
    model.load(e3_weights)
except Exception as e:
    print(f"Failed to load E3 weights: {e}")

# 2. Verify Tensor Shapes (Forward pass)
print("\nTesting forward pass on dummy tensor...")
dummy_input = torch.zeros(1, 3, 1536, 1536)
model.model.eval()
with torch.no_grad():
    outputs = model.model(dummy_input)

# In eval mode, outputs for a 4-head detector should be different.
# Let's inspect the shapes
print(f"Output type: {type(outputs)}")
if isinstance(outputs, (list, tuple)):
    for i, out in enumerate(outputs):
        if hasattr(out, 'shape'):
            print(f"Output {i} shape: {out.shape}")
        elif isinstance(out, (list, tuple)):
            print(f"Output {i} is a {type(out)} of length {len(out)}")
            for j, t in enumerate(out):
                if hasattr(t, 'shape'):
                    print(f"  - item {j} shape: {t.shape}")
        else:
            print(f"Output {i} type: {type(out)}")
else:
    print(f"Output shape: {outputs.shape}")

# 3. Class Count Verification
print(f"\nNumber of classes: {model.model.yaml['nc']}")

# 4. Parameter Count
total_params = sum(p.numel() for p in model.model.parameters())
print(f"Total Parameters: {total_params / 1e6:.2f} M")

# 5. Checkpoint saving/loading
print("\nTesting checkpoint save/load...")
model.save("scratch/test_p2.pt")
model2 = YOLO("scratch/test_p2.pt")
print("Successfully saved and loaded custom P2 checkpoint.")

# 6. Test inference
print("\nTesting inference API on dummy image...")
import numpy as np
dummy_img = np.zeros((1536, 1536, 3), dtype=np.uint8)
res = model2.predict(dummy_img, imgsz=1536, verbose=False)
print("Inference completed successfully.")
if len(res) > 0:
    print(f"Detected {len(res[0].boxes)} boxes.")
