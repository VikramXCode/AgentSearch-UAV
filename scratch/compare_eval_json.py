import sys, json
from pathlib import Path
from ultralytics import YOLO
sys.path.insert(0, str(Path.cwd()))
from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, CLASS_NAMES

# Generate YOLO val json
model = YOLO('runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt')
res = model.val(data="scratch/visdrone_10.yaml", imgsz=1536, batch=4, split='val', save_json=True, project="scratch", name="val_10", exist_ok=True, verbose=False)

# Load the JSON
with open("runs/detect/scratch/val_10/predictions.json", "r") as f:
    val_preds = json.load(f)

# The JSON is in COCO format: {"image_id": "image_name_without_ext", "category_id": ..., "bbox": [x, y, w, h], "score": ...}
# Wait, Ultralytics predictions.json uses `image_id` as the string stem (VisDrone image name).

preds_for_evaluator = {}
for p in val_preds:
    img_name = p["image_id"] + ".jpg" # assuming jpg
    if img_name not in preds_for_evaluator:
        preds_for_evaluator[img_name] = []
    
    x, y, w, h = p["bbox"]
    preds_for_evaluator[img_name].append({
        "class_id": p["category_id"] - 1, # COCO categories are 1-indexed? YOLO uses 0-indexed in its model but maybe val json maps it? No, VisDrone is 0-9 in YAML. Let's check!
        "label": CLASS_NAMES[p["category_id"]], # if category_id is 0-indexed
        "confidence": p["score"],
        "bbox": [x, y, x+w, y+h],
        "source": "val"
    })

evaluator = ComprehensiveEvaluator(iou_threshold=0.50)
evaluator.ground_truths = {k: v for k, v in evaluator.ground_truths.items() if k in preds_for_evaluator}
metrics = evaluator.evaluate_predictions(preds_for_evaluator)
with open("scratch/eval_json_res.json", "w") as f: json.dump({"map50": metrics["overall"]["map50"]}, f)
