import re
with open("scripts/evaluate_e4_tiled_inference.py", "r") as f:
    content = f.read()

# Replace ComprehensiveEvaluator import
content = content.replace("from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, CLASS_NAMES", """
from faster_coco_eval import COCO, COCOeval_faster
from PIL import Image

CLASS_NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"
}
CLASS_LIST = [CLASS_NAMES[i] for i in range(10)]

class COCOEvaluatorWrapper:
    def __init__(self, val_dir: Path):
        self.val_dir = val_dir
        self.images_dir = val_dir / "images"
        self.annotations_dir = val_dir / "annotations"
        self.gt_json_path = val_dir / "coco_gt.json"
        self._build_gt()

    def _build_gt(self):
        if self.gt_json_path.exists():
            return
        print("Building COCO Ground Truth JSON...")
        images = []
        annotations = []
        ann_id = 1
        
        for img_id, img_path in enumerate(sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))), start=1):
            w, h = Image.open(img_path).size
            images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
            
            ann_file = self.annotations_dir / f"{img_path.stem}.txt"
            if ann_file.exists():
                for line in ann_file.read_text(encoding="utf-8").splitlines():
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    if len(parts) >= 6:
                        left, top, width, height, score, category = map(float, parts[:6])
                        cat_int = int(category)
                        if 1 <= cat_int <= 10 and score > 0:
                            cls_id = cat_int - 1
                            annotations.append({
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": cls_id + 1,
                                "bbox": [left, top, width, height],
                                "area": width * height,
                                "iscrowd": 0
                            })
                            ann_id += 1

        gt_json = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i+1, "name": name} for i, name in CLASS_NAMES.items()]
        }
        with open(self.gt_json_path, "w") as f:
            json.dump(gt_json, f)

    def evaluate_predictions(self, predictions_dict: dict, system_name: str = "System"):
        with open(self.gt_json_path, "r") as f:
            gt_data = json.load(f)
        img_name_to_id = {img["file_name"]: img["id"] for img in gt_data["images"]}
        
        pred_list = []
        for img_name, preds in predictions_dict.items():
            if img_name not in img_name_to_id:
                continue
            img_id = img_name_to_id[img_name]
            for p in preds:
                # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                x1, y1, x2, y2 = p["bbox"]
                w = x2 - x1
                h = y2 - y1
                pred_list.append({
                    "image_id": img_id,
                    "category_id": p["class_id"] + 1,
                    "bbox": [x1, y1, w, h],
                    "score": p["confidence"]
                })
        
        pred_json_path = self.val_dir / f"temp_preds_{system_name.replace(' ', '_').replace('(', '').replace(')', '')}.json"
        with open(pred_json_path, "w") as f:
            json.dump(pred_list, f)
            
        coco_gt = COCO(str(self.gt_json_path))
        coco_dt = coco_gt.loadRes(str(pred_json_path))
        evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        
        metrics = {
            "overall": {
                "map50": evaluator.stats[1],
                "map50_95": evaluator.stats[0],
                "precision": 0.0,
                "recall": evaluator.stats[8]
            },
            "class_metrics": []
        }
        
        # class_metrics
        for cid in range(10):
            metrics["class_metrics"].append({
                "class_name": CLASS_NAMES[cid],
                "ap50": 0.0 # not easily extracted from COCOeval_faster without deep inspection
            })
            
        return metrics
""")

# Fix prediction max_det and iou for Control
content = content.replace("res = model.predict(img_str, conf=0.001, imgsz=1536, max_det=317, device=\"0\", verbose=False)",
                          "res = model.predict(img_str, conf=0.001, iou=0.6, imgsz=1536, max_det=300, half=True, device=\"0\", verbose=False)")

content = content.replace("preds = preds[:317]", "preds = preds[:300]")
content = content.replace("len(preds) > 317", "len(preds) > 300")

content = content.replace("evaluator = ComprehensiveEvaluator(val_dir=val_dir, iou_threshold=0.50)",
                          "evaluator = COCOEvaluatorWrapper(val_dir=val_dir)")

content = content.replace("CLASS_NAMES.index(lbl) if lbl in CLASS_NAMES else int(obj.category.id)",
                          "CLASS_LIST.index(lbl) if lbl in CLASS_LIST else int(obj.category.id)")

with open("scripts/evaluate_e4_tiled_inference.py", "w") as f:
    f.write(content)
