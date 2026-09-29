import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, CLASS_NAMES

with open("runs/detect/scratch/val_10/predictions.json", "r") as f:
    val_preds = json.load(f)

preds_for_evaluator = {}
for p in val_preds:
    img_name = p["image_id"] + ".jpg"
    if img_name not in preds_for_evaluator:
        preds_for_evaluator[img_name] = []
    
    x, y, w, h = p["bbox"]
    preds_for_evaluator[img_name].append({
        "class_id": p["category_id"] - 1, # wait, YOLO val saves category_id 0-9? Or 1-10? 
        "label": CLASS_NAMES[p["category_id"] - 1],
        "confidence": p["score"],
        "bbox": [x, y, x+w, y+h],
        "source": "val"
    })

evaluator = ComprehensiveEvaluator(iou_threshold=0.50)
evaluator.ground_truths = {k: v for k, v in evaluator.ground_truths.items() if k in preds_for_evaluator}
metrics = evaluator.evaluate_predictions(preds_for_evaluator)
print("ComprehensiveEvaluator on val JSON mAP50:", metrics["overall"]["map50"])
