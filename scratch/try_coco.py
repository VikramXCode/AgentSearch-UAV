import sys
from pathlib import Path
from ultralytics.models.yolo.detect.val import DetectionValidator

validator = DetectionValidator(args={"data": "configs/visdrone.yaml", "imgsz": 1536})
validator.data = validator.args.data
dataloader = validator.get_dataloader(validator.data["path"], 4)
validator.init_metrics(dataloader)

print("COCO JSON Path:", validator.jdict if validator.training else validator.save_dir)
