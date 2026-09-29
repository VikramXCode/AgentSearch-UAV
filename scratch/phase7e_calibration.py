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
            if len(parts) >= 6:
                if int(parts[5]) in [1, 2]: # pedestrian, people
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
    print("--- PHASE 7E: ADAPTIVE TRIGGER CALIBRATION ---")
    all_images = sorted(glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg")))
    dev_images = all_images[:50] # Phase 7C subset
    
    registry = ModelRegistry()
    e3_adapter = registry.get_adapter("specialist_e3")
    sahi = AdaptiveSAHI(default_slice_width=640, default_slice_height=640, default_overlap=0.20)
    
    dev_data = []
    
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
            "gt_count": len(gt_boxes),
            "features": feats,
            "base": {"tp": btp, "fp": bfp, "fn": bfn, "time": btime},
            "sahi": {"tp": stp, "fp": sfp, "fn": sfn, "time": stime}
        })
        print(f"DEV [{idx+1}/50] processed.")

    # Objective evaluation
    # Compute base total F1 and SAHI total F1, or just define beneficial per image
    total_btime = sum(d["base"]["time"] for d in dev_data)
    total_stime = sum(d["sahi"]["time"] for d in dev_data)
    
    for d in dev_data:
        bf1 = get_f1(d["base"]["tp"], d["base"]["fp"], d["base"]["fn"])
        sf1 = get_f1(d["sahi"]["tp"], d["sahi"]["fp"], d["sahi"]["fn"])
        d["sahi_beneficial"] = sf1 > bf1
    
    beneficial_count = sum(1 for d in dev_data if d["sahi_beneficial"])
    print(f"Total SAHI-beneficial images in dev: {beneficial_count}/50")

    # Evaluate a trigger logic
    def evaluate_trigger(trigger_fn):
        tp, fp, fn = 0, 0, 0
        trigger_count = 0
        correct_triggers = 0
        missed_beneficial = 0
        unnecessary_triggers = 0
        total_time = 0
        
        for d in dev_data:
            triggered = trigger_fn(d["features"])
            if triggered:
                trigger_count += 1
                tp += d["sahi"]["tp"]; fp += d["sahi"]["fp"]; fn += d["sahi"]["fn"]
                total_time += d["sahi"]["time"] + d["base"]["time"]
                if d["sahi_beneficial"]: correct_triggers += 1
                else: unnecessary_triggers += 1
            else:
                tp += d["base"]["tp"]; fp += d["base"]["fp"]; fn += d["base"]["fn"]
                total_time += d["base"]["time"]
                if d["sahi_beneficial"]: missed_beneficial += 1
                
        p, r = get_p_r(tp, fp, fn)
        f1 = get_f1(tp, fp, fn)
        return {
            "trigger_rate": trigger_count / len(dev_data),
            "sahi_executions": trigger_count,
            "unnecessary_sahi": unnecessary_triggers,
            "captured_beneficial": correct_triggers,
            "missed_beneficial": missed_beneficial,
            "tp": tp, "fp": fp, "fn": fn,
            "p": p, "r": r, "f1": f1,
            "time": total_time
        }

    triggers = {
        "A_count_gt_10": lambda f: f["count"] > 10,
        "A_count_gt_20": lambda f: f["count"] > 20,
        "A_count_gt_30": lambda f: f["count"] > 30,
        "B_tiny_gt_0.3": lambda f: f["frac_1024"] > 0.3,
        "B_tiny_gt_0.5": lambda f: f["frac_1024"] > 0.5,
        "C_density_gt_10": lambda f: f["density"] > 10,
        "C_density_gt_20": lambda f: f["density"] > 20,
        "D_cnt20_tiny0.4": lambda f: f["count"] > 20 and f["frac_1024"] > 0.4,
        "E_dens15_tiny0.4": lambda f: f["density"] > 15 and f["frac_1024"] > 0.4,
        "E_dens20_tiny0.4": lambda f: f["density"] > 20 and f["frac_1024"] > 0.4,
        "F_dens20_tiny0.4_conf_lt_0.5": lambda f: f["density"] > 20 and f["frac_1024"] > 0.4 and f["med_conf"] < 0.5,
    }
    
    print("--- TRIGGER CANDIDATES ON DEV ---")
    best_trigger_name = None
    best_f1 = -1
    best_res = None
    
    for name, fn in triggers.items():
        res = evaluate_trigger(fn)
        print(f"{name}: Rate={res['trigger_rate']:.2f}, F1={res['f1']:.3f}, P={res['p']:.3f}, R={res['r']:.3f}, Unnecessary={res['unnecessary_sahi']}")
        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            best_trigger_name = name
            best_res = res
            
    print(f"\nSELECTED TRIGGER: {best_trigger_name}")
    
    # We will just manually set the best policy here or save it.
    policy = {
        "trigger": best_trigger_name,
        "features": ["density", "frac_1024"], # example
        "thresholds": {"density_min": 20, "frac_1024_min": 0.4},
        "dev_rate": best_res["trigger_rate"],
        "estimated_compute_multiplier": best_res["time"] / total_btime if total_btime > 0 else 0,
        "rationale": "Maximized F1 while reducing unnecessary SAHI calls using interpretable density and tiny box fraction."
    }
    
    with open("scratch/adaptive_sahi_policy_7e.json", "w") as f:
        json.dump(policy, f, indent=2)

    # NOW EVALUATE ON HELD-OUT 7D DATA!
    print("--- HELD-OUT EVALUATION ON 7D ---")
    with open("scratch/calibration_7d.json", "r") as f:
        heldout = json.load(f)["images"]
        
    def heldout_trigger(f):
        # implement the winning trigger logic directly.
        # we will map the string to the lambda
        cnt = f["count"]
        frac = f["frac_1024"]
        dens = cnt / ((f["width"] if "width" in f else 1920) * (f["height"] if "height" in f else 1080) / 1000000.0)
        conf = f["med_conf"]
        if best_trigger_name == "D_cnt20_tiny0.4": return cnt > 20 and frac > 0.4
        elif best_trigger_name == "E_dens15_tiny0.4": return dens > 15 and frac > 0.4
        elif best_trigger_name == "E_dens20_tiny0.4": return dens > 20 and frac > 0.4
        elif best_trigger_name == "A_count_gt_20": return cnt > 20
        elif best_trigger_name == "A_count_gt_30": return cnt > 30
        elif best_trigger_name == "B_tiny_gt_0.5": return frac > 0.5
        elif best_trigger_name == "F_dens20_tiny0.4_conf_lt_0.5": return dens > 20 and frac > 0.4 and conf < 0.5
        elif best_trigger_name == "C_density_gt_20": return dens > 20
        else: return dens > 20 and frac > 0.4 # fallback
        
    ad_tp, ad_fp, ad_fn = 0, 0, 0
    ad_time = 0
    trigger_count = 0
    
    b_tp, b_fp, b_fn, b_time = 0, 0, 0, 0
    s_tp, s_fp, s_fn, s_time = 0, 0, 0, 0
    
    for d in heldout:
        cnt = d["candidate_stats"]["count"]
        # In 7D, we only recorded resolution in 'res', so dens:
        w, h = d["res"]
        dens = cnt / (w * h / 1000000.0)
        d["candidate_stats"]["width"] = w
        d["candidate_stats"]["height"] = h
        d["candidate_stats"]["density"] = dens
        
        b_tp += d["base"]["tp"]; b_fp += d["base"]["fp"]; b_fn += d["base"]["fn"]; b_time += d["base"]["time"]
        s_tp += d["sahi_c"]["tp"]; s_fp += d["sahi_c"]["fp"]; s_fn += d["sahi_c"]["fn"]; s_time += d["sahi_c"]["time"]
        
        triggered = heldout_trigger(d["candidate_stats"])
        if triggered:
            trigger_count += 1
            ad_tp += d["sahi_c"]["tp"]; ad_fp += d["sahi_c"]["fp"]; ad_fn += d["sahi_c"]["fn"]
            ad_time += d["sahi_c"]["time"] + d["base"]["time"]
        else:
            ad_tp += d["base"]["tp"]; ad_fp += d["base"]["fp"]; ad_fn += d["base"]["fn"]
            ad_time += d["base"]["time"]

    print("Always E3: F1={:.3f} (Time={:.1f}s)".format(get_f1(b_tp, b_fp, b_fn), b_time))
    print("Always SAHI: F1={:.3f} (Time={:.1f}s)".format(get_f1(s_tp, s_fp, s_fn), s_time))
    print("Adaptive SAHI: F1={:.3f} (Time={:.1f}s, Triggers={}/50)".format(get_f1(ad_tp, ad_fp, ad_fn), ad_time, trigger_count))
    print(f"Adaptive Stats: TP={ad_tp} FP={ad_fp} FN={ad_fn} P={get_p_r(ad_tp, ad_fp, ad_fn)[0]:.3f} R={get_p_r(ad_tp, ad_fp, ad_fn)[1]:.3f}")

if __name__ == "__main__":
    run()
