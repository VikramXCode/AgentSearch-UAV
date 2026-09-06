import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
from PIL import Image

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from models.yolo_world import YOLOWorldDetector
from models.sahi_engine import SAHIEngine
from models.enhanced_detector import SuperResolutionDetector
from models.optimized_detection_pipeline import OptimizedDetectionPipeline
from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, CLASS_NAMES

test_img = str(PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val" / "images" / "0000001_02999_d_0000005.jpg")

print(f"Testing on image: {test_img}")

# 1. Baseline YOLO-World
print("\n--- Testing System 1: Baseline YOLO-World ---")
base_model = YOLO("weights/yolov8s-world.pt")
if hasattr(base_model, "set_classes"):
    base_model.set_classes(CLASS_NAMES)
res1 = base_model.predict(test_img, conf=0.25, verbose=False)
print(f"System 1 detections: {len(res1[0].boxes)}")

# 2. Fine-Tuned YOLO-World
print("\n--- Testing System 2: Fine-Tuned YOLO-World ---")
ft_model = YOLO("weights/best.pt")
res2 = ft_model.predict(test_img, conf=0.25, verbose=False)
print(f"System 2 detections: {len(res2[0].boxes)}")

# 3. SAHI Pipeline
print("\n--- Testing System 3: SAHI Pipeline ---")
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="weights/best.pt",
    confidence_threshold=0.25,
    device="cpu"
)
sahi_res = get_sliced_prediction(
    test_img,
    sahi_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    verbose=0
)
print(f"System 3 detections: {len(sahi_res.object_prediction_list)}")

# 4. Super-Resolution Pipeline
print("\n--- Testing System 4: Super-Resolution Pipeline ---")
from models.super_resolution import SuperResolutionEngine
sr_engine = SuperResolutionEngine()
upscaled_path = sr_engine.upscale(test_img, scale=2)
res4 = ft_model.predict(upscaled_path, conf=0.25, verbose=False)
boxes4 = []
for box in res4[0].boxes:
    b = box.xyxy[0].tolist()
    # scale back to original
    boxes4.append([c * 0.5 for c in b])
print(f"System 4 detections: {len(boxes4)}")

# 5. AgentSearch-UAV Multi-Agent Pipeline
print("\n--- Testing System 5: AgentSearch-UAV Multi-Agent Pipeline ---")
multi_agent = OptimizedDetectionPipeline(
    model_path="weights/best.pt",
    config_path="configs/detection_config.json"
)
res5 = multi_agent.detect_image(test_img)
print(f"System 5 detections: {len(res5.filtered_detections)}")

print("\nAll 5 systems initialized and executed successfully!")
