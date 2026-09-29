import sys, json
from pathlib import Path
from ultralytics import YOLO
sys.path.insert(0, str(Path.cwd()))
from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, CLASS_NAMES

model = YOLO('runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt')
images_dir = Path("datasets/VisDrone2019/VisDrone2019-DET-val/images")
images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))[:10]

preds = {}
for img in images:
    res = model.predict(str(img), conf=0.001, iou=0.6, imgsz=1536, max_det=317, half=True, verbose=False)
    boxes = []
    if res and len(res[0].boxes) > 0:
        for box in res[0].boxes:
            cid = int(box.cls)
            if 0 <= cid < len(CLASS_NAMES):
                boxes.append({
                    "class_id": cid,
                    "label": CLASS_NAMES[cid],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),
                    "source": "control"
                })
    preds[img.name] = boxes

evaluator = ComprehensiveEvaluator(iou_threshold=0.50)
evaluator.ground_truths = {k: v for k, v in evaluator.ground_truths.items() if k in preds}
metrics = evaluator.evaluate_predictions(preds)
with open('scratch/compare_eval.json', 'w') as f:
    json.dump({"map50": metrics["overall"]["map50"]}, f)
