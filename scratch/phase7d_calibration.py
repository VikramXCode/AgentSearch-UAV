import os
import glob
import time
import json
import torch
import torchvision.ops as ops
from PIL import Image

from v2.models.model_registry import ModelRegistry
from v2.schemas.state import QuerySpec, MediaMetadata
from utils.adaptive_sahi import AdaptiveSAHI
from models.schemas import Detection
from models.detection_config import DEFAULT_CONFIG
from models.enhanced_postprocessor import EnhancedPostProcessor

# Configuration
VAL_IMAGES_DIR = "datasets/VisDrone2019/VisDrone2019-DET-val/images"
VAL_ANNO_DIR = "datasets/VisDrone2019/VisDrone2019-DET-val/annotations"
TARGET_CLASS = "person" # map to VisDrone categories 1 (pedestrian) and 2 (people)
IOU_THRESHOLD = 0.5
SIZE_BINS = {
    "small": (0, 1024),          # < 32x32
    "medium": (1024, 9216),      # 32x32 to 96x96
    "large": (9216, float('inf'))# > 96x96
}

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0: return 0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

def get_size_bin(area):
    for b, (low, high) in SIZE_BINS.items():
        if low <= area < high: return b
    return "unknown"

def parse_annotations(anno_path):
    gt_boxes = []
    if not os.path.exists(anno_path): return gt_boxes
    with open(anno_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                category = int(parts[5])
                if category in [1, 2]:
                    left, top, width, height = map(int, parts[:4])
                    gt_boxes.append({
                        "bbox": [left, top, left + width, top + height],
                        "area": width * height,
                    })
    return gt_boxes

def match_predictions(preds, gt_boxes):
    preds = sorted(preds, key=lambda x: x[4], reverse=True)
    tp, fp = 0, 0
    matched_gt_indices = set()
    for p in preds:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt_indices: continue
            iou = compute_iou(p[:4], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= IOU_THRESHOLD:
            tp += 1
            matched_gt_indices.add(best_gt_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt_indices)
    return tp, fp, fn, matched_gt_indices

def minimal_nms(detections, iou_threshold=0.45):
    if not detections:
        return []
    boxes = torch.tensor([d.bbox for d in detections], dtype=torch.float32)
    scores = torch.tensor([d.confidence for d in detections], dtype=torch.float32)
    keep_idx = ops.nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep_idx]

def run_calibration():
    print("--- PHASE 7D: SAHI EFFECT ISOLATION ---")
    all_images = sorted(glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg")))
    sample_images = all_images[50:100] # HELD-OUT [50:100]
    
    registry = ModelRegistry()
    e3_adapter = registry.get_adapter("specialist_e3")
    sahi = AdaptiveSAHI(default_slice_width=640, default_slice_height=640, default_overlap=0.20)
    
    def sahi_detect_fn(patches):
        results = e3_adapter.model(patches, verbose=False)
        batch_dets = []
        for r in results:
            dets = []
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = r.names[cls_id]
                dets.append(Detection(label=label, confidence=conf, bbox=[x1, y1, x2, y2]))
            batch_dets.append(dets)
        return batch_dets

    results = {
        "config": {
            "checkpoint": "runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt",
            "sahi": {"slice": 640, "overlap": 0.2},
            "iou_threshold": IOU_THRESHOLD,
            "target": TARGET_CLASS
        },
        "aggregate": {
            "base": {"tp": 0, "fp": 0, "fn": 0, "runtime": 0},
            "sahi_b": {"tp": 0, "fp": 0, "fn": 0, "runtime": 0}, # Enhanced NMS
            "sahi_c": {"tp": 0, "fp": 0, "fn": 0, "runtime": 0}, # Minimal NMS
        },
        "small_objects": {
            "base": {"tp": 0, "fn": 0},
            "sahi_b": {"tp": 0, "fn": 0},
            "sahi_c": {"tp": 0, "fn": 0}
        },
        "images": []
    }
    
    for idx, img_path in enumerate(sample_images):
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        anno_name = img_name.replace(".jpg", ".txt")
        anno_path = os.path.join(VAL_ANNO_DIR, anno_name)
        gt_boxes = parse_annotations(anno_path)
        
        # A. BASELINE E3
        query = QuerySpec(target=TARGET_CLASS)
        meta = MediaMetadata(resolution=[w, h])
        
        start_t = time.time()
        base_cands = e3_adapter.detect(img, query, meta)
        base_time = time.time() - start_t
        base_preds = [[c.bbox[0], c.bbox[1], c.bbox[2], c.bbox[3], c.confidence] for c in base_cands]
        btp, bfp, bfn, base_matched_indices = match_predictions(base_preds, gt_boxes)
        
        # Candidate Stats
        c_areas = [(c.bbox[2]-c.bbox[0])*(c.bbox[3]-c.bbox[1]) for c in base_cands]
        c_confs = [c.confidence for c in base_cands]
        med_area = sorted(c_areas)[len(c_areas)//2] if c_areas else 0.0
        max_conf = max(c_confs) if c_confs else 0.0
        med_conf = sorted(c_confs)[len(c_confs)//2] if c_confs else 0.0
        frac_1024 = sum(1 for a in c_areas if a < 1024) / len(c_areas) if c_areas else 0.0
        frac_2048 = sum(1 for a in c_areas if a < 2048) / len(c_areas) if c_areas else 0.0

        # SAHI GLOBAL DETECTIONS
        start_t = time.time()
        global_detections = sahi.run_sliced_inference(img, sahi_detect_fn, strategy="standard")
        sahi_raw_time = time.time() - start_t
        
        # B. SAHI + ENHANCED NMS
        start_t = time.time()
        try:
            merged_b = EnhancedPostProcessor.apply_nms(
                global_detections, config=DEFAULT_CONFIG, image_width=w, image_height=h
            )
        except Exception:
            merged_b = global_detections
        time_b = sahi_raw_time + (time.time() - start_t)
        preds_b = [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence] for d in merged_b if d.label in [TARGET_CLASS, "pedestrian", "people"]]
        tp_b, fp_b, fn_b, matched_b = match_predictions(preds_b, gt_boxes)
        
        # C. SAHI + MINIMAL NMS
        start_t = time.time()
        # filter valid labels first
        valid_global = [d for d in global_detections if d.label in [TARGET_CLASS, "pedestrian", "people"]]
        merged_c = minimal_nms(valid_global, iou_threshold=0.45)
        time_c = sahi_raw_time + (time.time() - start_t)
        preds_c = [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence] for d in merged_c]
        tp_c, fp_c, fn_c, matched_c = match_predictions(preds_c, gt_boxes)
        
        # Size stats
        for i, gt in enumerate(gt_boxes):
            sz = get_size_bin(gt["area"])
            if sz == "small":
                if i in base_matched_indices: results["small_objects"]["base"]["tp"] += 1
                else: results["small_objects"]["base"]["fn"] += 1
                
                if i in matched_b: results["small_objects"]["sahi_b"]["tp"] += 1
                else: results["small_objects"]["sahi_b"]["fn"] += 1
                
                if i in matched_c: results["small_objects"]["sahi_c"]["tp"] += 1
                else: results["small_objects"]["sahi_c"]["fn"] += 1

        img_res = {
            "name": img_name,
            "res": [w, h],
            "gt_count": len(gt_boxes),
            "base": {"tp": btp, "fp": bfp, "fn": bfn, "time": base_time},
            "sahi_b": {"tp": tp_b, "fp": fp_b, "fn": fn_b, "time": time_b},
            "sahi_c": {"tp": tp_c, "fp": fp_c, "fn": fn_c, "time": time_c},
            "candidate_stats": {
                "count": len(base_cands),
                "med_area": med_area,
                "max_conf": max_conf,
                "med_conf": med_conf,
                "frac_1024": frac_1024,
                "frac_2048": frac_2048
            }
        }
        results["images"].append(img_res)
        
        # Aggregate
        results["aggregate"]["base"]["tp"] += btp
        results["aggregate"]["base"]["fp"] += bfp
        results["aggregate"]["base"]["fn"] += bfn
        results["aggregate"]["base"]["runtime"] += base_time
        
        results["aggregate"]["sahi_b"]["tp"] += tp_b
        results["aggregate"]["sahi_b"]["fp"] += fp_b
        results["aggregate"]["sahi_b"]["fn"] += fn_b
        results["aggregate"]["sahi_b"]["runtime"] += time_b
        
        results["aggregate"]["sahi_c"]["tp"] += tp_c
        results["aggregate"]["sahi_c"]["fp"] += fp_c
        results["aggregate"]["sahi_c"]["fn"] += fn_c
        results["aggregate"]["sahi_c"]["runtime"] += time_c
        
        print(f"[{idx+1}/50] {img_name} | GT: {len(gt_boxes)} | Base F1: {2*btp/(2*btp+bfp+bfn) if (2*btp+bfp+bfn)>0 else 0:.3f} | B F1: {2*tp_b/(2*tp_b+fp_b+fn_b) if (2*tp_b+fp_b+fn_b)>0 else 0:.3f} | C F1: {2*tp_c/(2*tp_c+fp_c+fn_c) if (2*tp_c+fp_c+fn_c)>0 else 0:.3f}")

    with open("scratch/calibration_7d.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to scratch/calibration_7d.json")

if __name__ == "__main__":
    run_calibration()
