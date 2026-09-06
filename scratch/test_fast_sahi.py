import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
weights_path = PROJECT_ROOT / "weights" / "best.pt"
model = YOLO(str(weights_path))

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]
model.set_classes(CLASS_NAMES)

img_path = PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val" / "images" / "0000001_02999_d_0000005.jpg"
img = Image.open(img_path)
w, h = img.size

# Slicing
slice_w, slice_h = 960, 960
overlap = 0.2
step_x = int(slice_w * (1 - overlap))
step_y = int(slice_h * (1 - overlap))

patches = []
offsets = []

for y in range(0, h, step_y):
    for x in range(0, w, step_x):
        x1 = min(x, max(0, w - slice_w))
        y1 = min(y, max(0, h - slice_h))
        x2 = min(x1 + slice_w, w)
        y2 = min(y1 + slice_h, h)
        patch = img.crop((x1, y1, x2, y2))
        patches.append(patch)
        offsets.append((x1, y1))

print(f"Image {w}x{h} -> {len(patches)} slices")

# Time batched inference
t0 = time.time()
results = model.predict(source=patches, batch=len(patches), imgsz=960, conf=0.25, verbose=False, device="cpu")
t1 = time.time()

total_boxes = 0
for r, (ox, oy) in zip(results, offsets):
    if r.boxes is not None:
        total_boxes += len(r.boxes)

print(f"Batched slice inference: {t1 - t0:.3f}s for {len(patches)} slices (found {total_boxes} raw boxes)")
