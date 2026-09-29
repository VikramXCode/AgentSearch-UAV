import sys, json
from pathlib import Path
from PIL import Image
from faster_coco_eval import COCO, COCOeval_faster

# 1. Build GT JSON
CLASS_NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"
}

val_dir = Path("datasets/VisDrone2019/VisDrone2019-DET-val")
images_dir = val_dir / "images"
annotations_dir = val_dir / "annotations"

images = []
annotations = []
ann_id = 0

for img_id, img_path in enumerate(sorted(images_dir.glob("*.jpg"))[:10]):
    w, h = Image.open(img_path).size
    images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
    
    ann_file = annotations_dir / f"{img_path.stem}.txt"
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
                        "category_id": cls_id + 1, # 1-indexed for COCO
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

with open("scratch/gt_10.json", "w") as f:
    json.dump(gt_json, f)

# 2. Build Pred JSON from YOLO output
with open("runs/detect/scratch/val_10/predictions.json", "r") as f:
    val_preds = json.load(f)

img_name_to_id = {img["file_name"]: img["id"] for img in images}

pred_json = []
for p in val_preds:
    img_name = p["image_id"] + ".jpg"
    if img_name in img_name_to_id:
        pred_json.append({
            "image_id": img_name_to_id[img_name],
            "category_id": p["category_id"], # already 1-10
            "bbox": p["bbox"],
            "score": p["score"]
        })
        
with open("scratch/pred_10.json", "w") as f:
    json.dump(pred_json, f)

coco_gt = COCO("scratch/gt_10.json")
coco_dt = coco_gt.loadRes("scratch/pred_10.json")
evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()
print("COCO mAP50:", evaluator.stats[1])
