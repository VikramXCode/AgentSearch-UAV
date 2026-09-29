from ultralytics import YOLO
import sys, json
model = YOLO('runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt')
res = model.val(data="scratch/visdrone_10.yaml", imgsz=1536, batch=4, split='val', verbose=False)
with open("scratch/val_10.json", "w") as f:
    json.dump({"p": res.box.mp, "r": res.box.mr, "map50": res.box.map50}, f)
