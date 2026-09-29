from ultralytics import YOLO
import torch
model = YOLO('runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt')
res_val = model.val(data='configs/visdrone.yaml', imgsz=1536, batch=4, split='val')
print("YOLO val mAP50:", res_val.box.map50)
