import os
import glob
import time
import json
import torch
from PIL import Image

from v2.models.model_registry import ModelRegistry
from v2.schemas.state import AgentStateV2, ActionType, QuerySpec, MediaMetadata
from v2.agents.planning_agent import PlanningAgentV2
from utils.adaptive_sahi import AdaptiveSAHI
from models.schemas import Detection
from models.detection_config import DEFAULT_CONFIG
from models.enhanced_postprocessor import EnhancedPostProcessor

# Configuration
VAL_IMAGES_DIR = "datasets/VisDrone2019/VisDrone2019-DET-val/images"
VAL_ANNO_DIR = "datasets/VisDrone2019/VisDrone2019-DET-val/annotations"
TARGET_CLASS = "person" # we will focus on pedestrian(1) and people(2) for GT matching
# Actually, let's map YOLO-World 'person' to VisDrone GT classes 1 (pedestrian) and 2 (people).
IOU_THRESHOLD = 0.5
SIZE_BINS = {
    "small": (0, 1024),          # < 32x32
    "medium": (1024, 9216),      # 32x32 to 96x96
    "large": (9216, float('inf'))# > 96x96
}

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

def get_size_bin(area):
    for b, (low, high) in SIZE_BINS.items():
        if low <= area < high:
            return b
    return "unknown"

def parse_annotations(anno_path):
    gt_boxes = []
    if not os.path.exists(anno_path):
        return gt_boxes
    with open(anno_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                category = int(parts[5])
                # In VisDrone, 1=pedestrian, 2=people, 3=bicycle, 4=car, etc.
                # We will map 'person' to category 1 and 2.
                if category in [1, 2]:
                    left, top, width, height = map(int, parts[:4])
                    # skip ignored regions (category 0 or truncations) if needed, but keeping simple
                    gt_boxes.append({
                        "bbox": [left, top, left + width, top + height],
                        "area": width * height,
                        "matched": False,
                        "matched_by_sahi": False
                    })
    return gt_boxes

def match_predictions(preds, gt_boxes):
    # preds: list of [x1, y1, x2, y2, conf]
    # gt_boxes: list of dicts with 'bbox' and 'matched' flag
    # Sort predictions by confidence
    preds = sorted(preds, key=lambda x: x[4], reverse=True)
    tp, fp = 0, 0
    matched_gt_indices = set()
    
    for p in preds:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt_indices:
                continue
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

def run_calibration():
    print("--- PHASE 7C: GT-BASED CALIBRATION ---")
    all_images = sorted(glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg")))
    sample_images = all_images[:50]
    
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
            "size_bins": SIZE_BINS,
            "target": TARGET_CLASS
        },
        "aggregate": {
            "base": {"tp": 0, "fp": 0, "fn": 0, "runtime": 0},
            "sahi": {"tp": 0, "fp": 0, "fn": 0, "runtime": 0},
        },
        "small_objects": {
            "base_detected": 0, "sahi_recovered": 0, "detected_by_both": 0, "missed_by_both": 0
        },
        "planner_triggers": {
            "sahi_triggers": 0,
            "sr_triggers": 0,
            "verify_triggers": 0
        },
        "images": []
    }
    
    planner = PlanningAgentV2()
    
    for idx, img_path in enumerate(sample_images):
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        anno_name = img_name.replace(".jpg", ".txt")
        anno_path = os.path.join(VAL_ANNO_DIR, anno_name)
        gt_boxes = parse_annotations(anno_path)
        
        # 1. BASELINE
        query = QuerySpec(target=TARGET_CLASS)
        meta = MediaMetadata(resolution=[w, h])
        
        start_t = time.time()
        base_cands = e3_adapter.detect(img, query, meta)
        base_time = time.time() - start_t
        
        base_preds = [[c.bbox[0], c.bbox[1], c.bbox[2], c.bbox[3], c.confidence] for c in base_cands]
        btp, bfp, bfn, base_matched_indices = match_predictions(base_preds, gt_boxes)
        
        # 2. SAHI
        start_t = time.time()
        global_detections = sahi.run_sliced_inference(img, sahi_detect_fn, strategy="standard")
        try:
            merged_detections = EnhancedPostProcessor.apply_nms(
                global_detections, config=DEFAULT_CONFIG, image_width=w, image_height=h
            )
        except Exception:
            merged_detections = global_detections
            
        sahi_time = time.time() - start_t
        sahi_preds = [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence] for d in merged_detections if d.label in [TARGET_CLASS, "pedestrian", "people"]]
        stp, sfp, sfn, sahi_matched_indices = match_predictions(sahi_preds, gt_boxes)
        
        # 3. STATS & SIZES
        added_tp = max(0, stp - btp)
        added_fp = max(0, sfp - bfp)
        
        # Size analysis for GT
        for i, gt in enumerate(gt_boxes):
            sz = get_size_bin(gt["area"])
            base_hit = i in base_matched_indices
            sahi_hit = i in sahi_matched_indices
            
            if sz == "small":
                if base_hit and sahi_hit: results["small_objects"]["detected_by_both"] += 1
                elif base_hit and not sahi_hit: results["small_objects"]["base_detected"] += 1
                elif not base_hit and sahi_hit: results["small_objects"]["sahi_recovered"] += 1
                else: results["small_objects"]["missed_by_both"] += 1
        
        # 4. PLANNER TRIGGERS
        state = AgentStateV2()
        state.query_spec.target = TARGET_CLASS
        state.media_metadata.resolution = [w, h]
        state.candidates = base_cands
        state.plan.history.append(ActionType.DETECT_SPECIALIST)
        
        action = planner.run(state)
        if action == ActionType.ENHANCE_SAHI: results["planner_triggers"]["sahi_triggers"] += 1
        elif action == ActionType.ENHANCE_SR: results["planner_triggers"]["sr_triggers"] += 1
        else: results["planner_triggers"]["verify_triggers"] += 1
        
        cands_areas = [(c.bbox[2]-c.bbox[0])*(c.bbox[3]-c.bbox[1]) for c in base_cands]
        med_area = sorted(cands_areas)[len(cands_areas)//2] if cands_areas else 0.0
        confs = [c.confidence for c in base_cands]
        max_conf = max(confs) if confs else 0.0
        
        img_res = {
            "name": img_name,
            "res": [w, h],
            "gt_count": len(gt_boxes),
            "base": {"count": len(base_cands), "tp": btp, "fp": bfp, "fn": bfn, "time": base_time},
            "sahi": {"count": len(sahi_preds), "tp": stp, "fp": sfp, "fn": sfn, "time": sahi_time},
            "planner_action": str(action),
            "stats": {"med_area": med_area, "max_conf": max_conf}
        }
        results["images"].append(img_res)
        
        # Aggregate
        results["aggregate"]["base"]["tp"] += btp
        results["aggregate"]["base"]["fp"] += bfp
        results["aggregate"]["base"]["fn"] += bfn
        results["aggregate"]["base"]["runtime"] += base_time
        
        results["aggregate"]["sahi"]["tp"] += stp
        results["aggregate"]["sahi"]["fp"] += sfp
        results["aggregate"]["sahi"]["fn"] += sfn
        results["aggregate"]["sahi"]["runtime"] += sahi_time
        
        print(f"[{idx+1}/50] {img_name} | GT: {len(gt_boxes)} | Base F1: {2*btp/(2*btp+bfp+bfn) if (2*btp+bfp+bfn)>0 else 0:.3f} | SAHI F1: {2*stp/(2*stp+sfp+sfn) if (2*stp+sfp+sfn)>0 else 0:.3f} | Planner: {action.name}")

    with open("scratch/calibration_7c.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to scratch/calibration_7c.json")

if __name__ == "__main__":
    run_calibration()
