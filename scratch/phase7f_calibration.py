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

# Configuration
VAL_IMAGES_DIR = "datasets/VisDrone2019/VisDrone2019-DET-val/images"
VAL_ANNO_DIR = "datasets/VisDrone2019/VisDrone2019-DET-val/annotations"
TARGET_CLASS = "person"
IOU_THRESHOLD = 0.5

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0: return 0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

def parse_annotations(anno_path):
    gt_boxes = []
    if not os.path.exists(anno_path): return gt_boxes
    with open(anno_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6 and int(parts[5]) in [1, 2]:
                left, top, width, height = map(int, parts[:4])
                gt_boxes.append({"bbox": [left, top, left + width, top + height]})
    return gt_boxes

def match_predictions(preds, gt_boxes):
    preds = sorted(preds, key=lambda x: x[4], reverse=True)
    tp, fp = 0, 0
    matched = set()
    for p in preds:
        best_iou, best_idx = 0, -1
        for i, gt in enumerate(gt_boxes):
            if i in matched: continue
            iou = compute_iou(p[:4], gt["bbox"])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= IOU_THRESHOLD:
            tp += 1
            matched.add(best_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn

def minimal_nms(detections, iou_threshold=0.45):
    if not detections: return []
    boxes = torch.tensor([d.bbox for d in detections], dtype=torch.float32)
    scores = torch.tensor([d.confidence for d in detections], dtype=torch.float32)
    keep_idx = ops.nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep_idx]

def get_f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * p * r / (p + r) if p + r > 0 else 0

def get_p_r(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return p, r

def extract_features(img, e3_adapter):
    w, h = img.size
    query = QuerySpec(target=TARGET_CLASS)
    meta = MediaMetadata(resolution=[w, h])
    
    start = time.time()
    cands = e3_adapter.detect(img, query, meta)
    btime = time.time() - start
    
    cnt = len(cands)
    areas = [(c.bbox[2]-c.bbox[0])*(c.bbox[3]-c.bbox[1]) for c in cands]
    confs = [c.confidence for c in cands]
    
    med_area = sorted(areas)[len(areas)//2] if areas else 0
    max_conf = max(confs) if confs else 0
    med_conf = sorted(confs)[len(confs)//2] if confs else 0
    frac_1024 = sum(1 for a in areas if a < 1024) / cnt if cnt > 0 else 0
    megapixels = (w * h) / 1000000.0
    density = cnt / megapixels if megapixels > 0 else 0
    
    return cands, {
        "count": cnt,
        "frac_1024": frac_1024,
        "med_area": med_area,
        "max_conf": max_conf,
        "med_conf": med_conf,
        "width": w, "height": h, "area": w*h,
        "density": density
    }, btime

def get_sahi_preds(img, sahi, e3_adapter):
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
    
    start = time.time()
    global_dets = sahi.run_sliced_inference(img, sahi_detect_fn, strategy="standard")
    valid_global = [d for d in global_dets if d.label in [TARGET_CLASS, "pedestrian", "people"]]
    merged = minimal_nms(valid_global, iou_threshold=0.45)
    stime = time.time() - start
    return merged, stime

def run():
    print("--- PHASE 7F: MULTI-SIGNAL ADAPTIVE SAHI POLICY STRESS TEST ---")
    all_images = sorted(glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg")))
    dev_images = all_images[:50]
    
    registry = ModelRegistry()
    e3_adapter = registry.get_adapter("specialist_e3")
    sahi = AdaptiveSAHI(default_slice_width=640, default_slice_height=640, default_overlap=0.20)
    
    dev_data = []
    
    # 1. PROCESS DEV SET
    for idx, img_path in enumerate(dev_images):
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        gt_boxes = parse_annotations(os.path.join(VAL_ANNO_DIR, img_name.replace(".jpg", ".txt")))
        
        cands, feats, btime = extract_features(img, e3_adapter)
        bpreds = [[c.bbox[0], c.bbox[1], c.bbox[2], c.bbox[3], c.confidence] for c in cands]
        btp, bfp, bfn = match_predictions(bpreds, gt_boxes)
        
        sahi_cands, stime = get_sahi_preds(img, sahi, e3_adapter)
        spreds = [[c.bbox[0], c.bbox[1], c.bbox[2], c.bbox[3], c.confidence] for c in sahi_cands]
        stp, sfp, sfn = match_predictions(spreds, gt_boxes)
        
        dev_data.append({
            "name": img_name,
            "features": feats,
            "base": {"tp": btp, "fp": bfp, "fn": bfn, "time": btime},
            "sahi": {"tp": stp, "fp": sfp, "fn": sfn, "time": stime}
        })
        print(f"DEV [{idx+1}/50] processed.")

    base_f1_dev = get_f1(
        sum(d["base"]["tp"] for d in dev_data),
        sum(d["base"]["fp"] for d in dev_data),
        sum(d["base"]["fn"] for d in dev_data)
    )

    # 2. GENERATE POLICIES
    dens_th = [5, 10, 15, 20, 30]
    cnt_th = [10, 20, 40, 60, 100]
    tiny_th = [0.25, 0.40, 0.50, 0.60, 0.75]
    conf_th = [0.30, 0.40, 0.50]
    
    policies = []
    
    # A: density > D
    for d in dens_th:
        policies.append({
            "name": f"A_dens_{d}",
            "fn": lambda f, d=d: f["density"] > d,
            "config": {"density_min": d},
            "features": ["density"],
            "type": "A"
        })
    # B: tiny_fraction > T
    for t in tiny_th:
        policies.append({
            "name": f"B_tiny_{t}",
            "fn": lambda f, t=t: f["frac_1024"] > t,
            "config": {"frac_1024_min": t},
            "features": ["frac_1024"],
            "type": "B"
        })
    # C: count > C
    for c in cnt_th:
        policies.append({
            "name": f"C_count_{c}",
            "fn": lambda f, c=c: f["count"] > c,
            "config": {"count_min": c},
            "features": ["count"],
            "type": "C"
        })
    # D: density > D AND tiny_fraction > T
    for d in dens_th:
        for t in tiny_th:
            policies.append({
                "name": f"D_dens_{d}_AND_tiny_{t}",
                "fn": lambda f, d=d, t=t: f["density"] > d and f["frac_1024"] > t,
                "config": {"density_min": d, "frac_1024_min": t},
                "features": ["density", "frac_1024"],
                "type": "D"
            })
    # E: density > D OR tiny_fraction > T
    for d in dens_th:
        for t in tiny_th:
            policies.append({
                "name": f"E_dens_{d}_OR_tiny_{t}",
                "fn": lambda f, d=d, t=t: f["density"] > d or f["frac_1024"] > t,
                "config": {"density_min": d, "frac_1024_min": t},
                "features": ["density", "frac_1024"],
                "type": "E"
            })
    # F: count > C AND tiny_fraction > T
    for c in cnt_th:
        for t in tiny_th:
            policies.append({
                "name": f"F_count_{c}_AND_tiny_{t}",
                "fn": lambda f, c=c, t=t: f["count"] > c and f["frac_1024"] > t,
                "config": {"count_min": c, "frac_1024_min": t},
                "features": ["count", "frac_1024"],
                "type": "F"
            })
    # G: density > D OR tiny_fraction > T OR med_conf < Conf
    for d in dens_th:
        for t in tiny_th:
            for conf in conf_th:
                policies.append({
                    "name": f"G_dens_{d}_OR_tiny_{t}_OR_conf_{conf}",
                    "fn": lambda f, d=d, t=t, c=conf: f["density"] > d or f["frac_1024"] > t or f["med_conf"] < c,
                    "config": {"density_min": d, "frac_1024_min": t, "med_conf_max": c},
                    "features": ["density", "frac_1024", "med_conf"],
                    "type": "G"
                })

    # 3. EVALUATE POLICIES ON DEV
    def evaluate_policy(p_fn):
        tp, fp, fn = 0, 0, 0
        triggers = 0
        for d in dev_data:
            if p_fn(d["features"]):
                triggers += 1
                tp += d["sahi"]["tp"]; fp += d["sahi"]["fp"]; fn += d["sahi"]["fn"]
            else:
                tp += d["base"]["tp"]; fp += d["base"]["fp"]; fn += d["base"]["fn"]
        return get_f1(tp, fp, fn), triggers / len(dev_data)

    results = []
    for p in policies:
        f1, rate = evaluate_policy(p["fn"])
        if f1 > base_f1_dev and rate < 1.0: # 1. Outperform Base E3, 2. Less SAHI than Always-SAHI
            results.append((f1, rate, p))

    # 4. SELECT BEST POLICY
    results.sort(key=lambda x: x[0], reverse=True) # Sort by F1 descending
    best_f1 = results[0][0]
    
    # 3. Among similarly performing policies (F1 within 0.01 of max), use less compute (lower rate)
    candidates = [r for r in results if r[0] >= best_f1 - 0.01]
    candidates.sort(key=lambda x: x[1]) # Sort by trigger rate ascending
    selected = candidates[0]
    
    sel_f1, sel_rate, sel_policy = selected
    print(f"\nSELECTED POLICY: {sel_policy['name']}")
    print(f"DEV F1: {sel_f1:.3f} | Rate: {sel_rate:.2f}")

    # Freeze policy to scratch/adaptive_sahi_policy_7f.json
    policy_output = {
        "name": sel_policy["name"],
        "type": sel_policy["type"],
        "features": sel_policy["features"],
        "thresholds": sel_policy["config"],
        "selection_criterion": "Maximized DEV F1, then minimized trigger rate for those within 0.01 F1 of maximum.",
        "dev_f1": sel_f1,
        "dev_rate": sel_rate
    }
    with open("scratch/adaptive_sahi_policy_7f.json", "w") as f:
        json.dump(policy_output, f, indent=2)

    # 5. HELD-OUT EVALUATION ON PHASE 7D
    with open("scratch/calibration_7d.json", "r") as f:
        heldout_data = json.load(f)["images"]

    metrics = {
        "base": {"tp": 0, "fp": 0, "fn": 0, "time": 0},
        "sahi": {"tp": 0, "fp": 0, "fn": 0, "time": 0},
        "phase7e": {"tp": 0, "fp": 0, "fn": 0, "time": 0, "triggers": 0},
        "phase7f": {"tp": 0, "fp": 0, "fn": 0, "time": 0, "triggers": 0}
    }
    
    for d in heldout_data:
        metrics["base"]["tp"] += d["base"]["tp"]; metrics["base"]["fp"] += d["base"]["fp"]; metrics["base"]["fn"] += d["base"]["fn"]
        metrics["base"]["time"] += d["base"]["time"]
        
        metrics["sahi"]["tp"] += d["sahi_c"]["tp"]; metrics["sahi"]["fp"] += d["sahi_c"]["fp"]; metrics["sahi"]["fn"] += d["sahi_c"]["fn"]
        metrics["sahi"]["time"] += d["sahi_c"]["time"]
        
        # reconstruct candidate_stats for 7D images
        cnt = d["candidate_stats"]["count"]
        w, h = d["res"]
        frac_1024 = d["candidate_stats"]["frac_1024"]
        med_conf = d["candidate_stats"]["med_conf"]
        density = cnt / (w * h / 1000000.0)
        
        feats = {
            "count": cnt, "density": density, "frac_1024": frac_1024, "med_conf": med_conf
        }
        
        # Phase 7E logic: density > 10
        if density > 10:
            metrics["phase7e"]["triggers"] += 1
            metrics["phase7e"]["tp"] += d["sahi_c"]["tp"]; metrics["phase7e"]["fp"] += d["sahi_c"]["fp"]; metrics["phase7e"]["fn"] += d["sahi_c"]["fn"]
            metrics["phase7e"]["time"] += d["sahi_c"]["time"] + d["base"]["time"]
        else:
            metrics["phase7e"]["tp"] += d["base"]["tp"]; metrics["phase7e"]["fp"] += d["base"]["fp"]; metrics["phase7e"]["fn"] += d["base"]["fn"]
            metrics["phase7e"]["time"] += d["base"]["time"]
            
        # Phase 7F logic
        if sel_policy["fn"](feats):
            metrics["phase7f"]["triggers"] += 1
            metrics["phase7f"]["tp"] += d["sahi_c"]["tp"]; metrics["phase7f"]["fp"] += d["sahi_c"]["fp"]; metrics["phase7f"]["fn"] += d["sahi_c"]["fn"]
            metrics["phase7f"]["time"] += d["sahi_c"]["time"] + d["base"]["time"]
        else:
            metrics["phase7f"]["tp"] += d["base"]["tp"]; metrics["phase7f"]["fp"] += d["base"]["fp"]; metrics["phase7f"]["fn"] += d["base"]["fn"]
            metrics["phase7f"]["time"] += d["base"]["time"]
            
    res_out = {}
    print("\n--- HELD-OUT RESULTS ON 7D ---")
    for k in metrics:
        tp, fp, fn = metrics[k]["tp"], metrics[k]["fp"], metrics[k]["fn"]
        f1 = get_f1(tp, fp, fn)
        p, r = get_p_r(tp, fp, fn)
        rate = metrics[k].get("triggers", 50) / 50.0
        print(f"{k.upper()}: F1={f1:.3f} | P={p:.3f} | R={r:.3f} | Rate={rate:.2f} | Time={metrics[k]['time']:.1f}s")
        res_out[k] = {"f1": f1, "p": p, "r": r, "rate": rate, "time": metrics[k]["time"]}
        
    with open("scratch/phase7f_results.json", "w") as f:
        json.dump(res_out, f, indent=2)

if __name__ == "__main__":
    run()
