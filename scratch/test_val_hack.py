from ultralytics.models.yolo.detect.val import DetectionValidator
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

validator = DetectionValidator(args={"data": "scratch/visdrone_10.yaml", "imgsz": 1536})
# We need to initialize the validator so it builds the dataloader and gets the ground truth COCO json!
validator.data = validator.args.data
validator.init_metrics(validator.get_dataloader(validator.data, 4))
print(validator.jdict)
