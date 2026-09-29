import json
import sys
from pathlib import Path
from PIL import Image

from v2.evaluation.visdrone_pairs import collect_visdrone_crops, generate_text_calibration_pairs
from v2.models.model_registry import ModelRegistry
from v2.models.semantic_adapter import CLIPEngineAdapter
from v2.evaluation.calibration import CalibrationRecord, GroundTruthLabel, ThresholdEvaluator

def calculate_iou(box1, box2):
    # box: (x1, y1, x2, y2)
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def evaluate_predictions(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1
    }

def run_experiment():
    print("=== PHASE 5F: VISDRONE DETECTOR EVIDENCE VS CLIP VERIFICATION ===")
    dataset_path = "datasets/VisDrone2019"
    crops = collect_visdrone_crops(dataset_path, max_per_class=20)
    pairs = generate_text_calibration_pairs(crops)
    
    # 1. RUN DETECTOR (E3) ON FULL IMAGES
    registry = ModelRegistry()
    try:
        e3_model = registry._load_model("specialist_e3")
    except Exception as e:
        print(f"Failed to load E3: {e}")
        return
        
    image_paths = list(set([c["image_path"] for c in crops]))
    print(f"Running E3 detection on {len(image_paths)} unique images...")
    
    crop_predictions = {} # map crop id to (pred_class_name, confidence)
    
    for img_path in image_paths:
        results = e3_model(img_path, verbose=False)
        if len(results) == 0:
            continue
            
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        names = r.names
        
        # Match crops from this image
        img_crops = [c for c in crops if c["image_path"] == img_path]
        for c in img_crops:
            crop_id = f"{c['image_id']}_{c['class_name']}_{c['bbox']}"
            best_iou = 0
            best_match = None
            
            for box, conf, cls in zip(boxes, confs, classes):
                iou = calculate_iou(c["bbox"], box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = (names[int(cls)], conf)
                    
            if best_iou > 0.5 and best_match:
                crop_predictions[crop_id] = best_match
            else:
                crop_predictions[crop_id] = (None, 0.0)
                
    # 2. RUN CLIP OR LOAD EXISTING SCORES
    print("Loading CLIP scores...")
    try:
        with open("scratch/visdrone_calibration_scores_5e.json", "r") as f:
            clip_scores_data = json.load(f)
            # map by query + "_" + candidate_crop_info
            clip_scores = {f"{r['query']}_{r['crop_id']}": r["score"] for r in clip_scores_data}
    except Exception as e:
        print(f"Failed to load CLIP scores: {e}")
        return
        
    # 3. EVALUATE ALL PAIRS
    y_true = []
    y_pred_e3 = []
    y_pred_clip = []
    y_pred_rule1 = [] # E3 AND CLIP
    y_pred_rule2 = [] # E3 OR CLIP
    
    # Let's use the balanced F1 threshold found in 5E for CLIP
    CLIP_THRESH = 0.219
    
    for pair in pairs:
        crop_id = f"{pair['crop']['image_id']}_{pair['crop']['class_name']}_{pair['crop']['bbox']}"
        # Fallback crop ID mapping to match JSON from 5E (where I just used image_id_class_name)
        fallback_crop_id = f"{pair['crop']['image_id']}_{pair['crop']['class_name']}"
        key = f"{pair['query']}_{fallback_crop_id}"
        clip_score = clip_scores.get(key, 0.0)
        
        e3_pred_class, e3_conf = crop_predictions.get(crop_id, (None, 0.0))
        
        is_match_gt = (pair["ground_truth"] == "MATCH")
        
        is_match_e3 = (e3_pred_class == pair["query"])
        is_match_clip = (clip_score >= CLIP_THRESH)
        
        y_true.append(is_match_gt)
        y_pred_e3.append(is_match_e3)
        y_pred_clip.append(is_match_clip)
        y_pred_rule1.append(is_match_e3 and is_match_clip)
        y_pred_rule2.append(is_match_e3 or is_match_clip)
        
    # Calculate metrics
    res_e3 = evaluate_predictions(y_true, y_pred_e3)
    res_clip = evaluate_predictions(y_true, y_pred_clip)
    res_rule1 = evaluate_predictions(y_true, y_pred_rule1)
    res_rule2 = evaluate_predictions(y_true, y_pred_rule2)
    
    # HARD NEGATIVES ONLY
    # A negative pair is hard_negative if pair["pair_type"] == "hard_negative"
    hn_indices = [i for i, p in enumerate(pairs) if p["pair_type"] == "hard_negative"]
    y_true_hn = [y_true[i] for i in hn_indices]
    y_pred_e3_hn = [y_pred_e3[i] for i in hn_indices]
    y_pred_clip_hn = [y_pred_clip[i] for i in hn_indices]
    y_pred_rule1_hn = [y_pred_rule1[i] for i in hn_indices]
    y_pred_rule2_hn = [y_pred_rule2[i] for i in hn_indices]

    res_e3_hn = evaluate_predictions(y_true_hn, y_pred_e3_hn)
    res_clip_hn = evaluate_predictions(y_true_hn, y_pred_clip_hn)
    res_rule1_hn = evaluate_predictions(y_true_hn, y_pred_rule1_hn)
    res_rule2_hn = evaluate_predictions(y_true_hn, y_pred_rule2_hn)
    
    # TRIVIAL NEGATIVES ONLY
    tn_indices = [i for i, p in enumerate(pairs) if p["pair_type"] == "trivial_negative"]
    y_true_tn = [y_true[i] for i in tn_indices]
    y_pred_e3_tn = [y_pred_e3[i] for i in tn_indices]
    y_pred_clip_tn = [y_pred_clip[i] for i in tn_indices]
    
    res_e3_tn = evaluate_predictions(y_true_tn, y_pred_e3_tn)
    res_clip_tn = evaluate_predictions(y_true_tn, y_pred_clip_tn)

    print("\n--- DETECTOR-ONLY (E3) RESULTS ---")
    print(json.dumps(res_e3, indent=2))
    
    print(f"\n--- CLIP-ONLY RESULTS (Threshold {CLIP_THRESH}) ---")
    print(json.dumps(res_clip, indent=2))
    
    print("\n--- COMBINED RULE 1: E3 AND CLIP ---")
    print(json.dumps(res_rule1, indent=2))
    
    print("\n--- COMBINED RULE 2: E3 OR CLIP ---")
    print(json.dumps(res_rule2, indent=2))
    
    print("\n--- HARD NEGATIVES ONLY (E3 vs CLIP False Positives) ---")
    print(f"E3 False Positives on Hard Negatives: {res_e3_hn['FP']} out of {len(hn_indices)}")
    print(f"CLIP False Positives on Hard Negatives: {res_clip_hn['FP']} out of {len(hn_indices)}")
    print(f"RULE 1 False Positives on Hard Negatives: {res_rule1_hn['FP']} out of {len(hn_indices)}")
    
    print("\n--- TRIVIAL NEGATIVES ONLY (E3 vs CLIP False Positives) ---")
    print(f"E3 False Positives on Trivial Negatives: {res_e3_tn['FP']} out of {len(tn_indices)}")
    print(f"CLIP False Positives on Trivial Negatives: {res_clip_tn['FP']} out of {len(tn_indices)}")
        
if __name__ == "__main__":
    run_experiment()
