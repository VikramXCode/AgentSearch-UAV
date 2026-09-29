with open("scripts/evaluate_e4_tiled_inference.py", "r") as f:
    content = f.read()

content = content.replace(
    'evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")',
    'evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")\n        img_ids = list(set(p["image_id"] for p in pred_list))\n        if img_ids:\n            evaluator.params.imgIds = img_ids'
)

with open("scripts/evaluate_e4_tiled_inference.py", "w") as f:
    f.write(content)
