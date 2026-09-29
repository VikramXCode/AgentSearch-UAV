from ultralytics import YOLO
import json

model = YOLO("runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt")
results = model.val(data="configs/visdrone.yaml", imgsz=1536, device="0", split="val")

with open("scratch/val_full_res.json", "w") as f:
    json.dump({
        "map50": results.results_dict.get("metrics/mAP50(B)")
    }, f)
print("Finished YOLO.val(). mAP50:", results.results_dict.get("metrics/mAP50(B)"))
