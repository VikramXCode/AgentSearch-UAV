import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from evaluation.comprehensive_evaluator import ComprehensiveEvaluator
evaluator = ComprehensiveEvaluator()
with open("runs/detect/scratch/val_10/predictions.json", "r") as f:
    val_preds = json.load(f)
imgs = set(p["image_id"] + ".jpg" for p in val_preds)
total = 0
for img in imgs:
    if img in evaluator.ground_truths:
        total += len(evaluator.ground_truths[img])
print("ComprehensiveEvaluator GTs for 10 images:", total)
