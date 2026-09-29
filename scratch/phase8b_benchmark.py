#!/usr/bin/env python3
"""
Phase 8B: Final Held-Out Detection Benchmark (Execution Script)
Evaluates E3 Base vs E3 + SAHI on the strictly untouched Phase 8 subset (images [100:548]).
"""

import os
import sys
import glob
import time
import json
import torch
import torchvision.ops as ops
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from utils.adaptive_sahi import AdaptiveSAHI
from models.schemas import Detection

CHECKPOINT_PATH = PROJECT_ROOT / "runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt"
VAL_DIR = PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val"
IMAGES_DIR = VAL_DIR / "images"

# Exact Benchmark Configuration
IMGSZ = 1536
CONF_THRESH = 0.25
NMS_IOU = 0.45
SAHI_SLICE = 640
SAHI_OVERLAP = 0.20
SAHI_NMS = 0.45

def minimal_nms(detections, iou_threshold=0.45):
    if not detections: return []
    boxes = torch.tensor([d.bbox for d in detections], dtype=torch.float32)
    scores = torch.tensor([d.confidence for d in detections], dtype=torch.float32)
    keep_idx = ops.nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep_idx]

def extract_preds_from_ultralytics(results):
    preds = []
    for r in results:
        boxes = r.boxes
        if boxes is None: continue
        for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            label = r.names[int(cls)]
            preds.append({
                "bbox": box.tolist(),
                "confidence": float(conf),
                "label": label
            })
    return preds

def run_benchmark():
    print("--- PHASE 8B FINAL EVALUATION BENCHMARK ---")
    
    # 1. Verify checkpoint
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"E3 Checkpoint missing: {CHECKPOINT_PATH}")
    print(f"Verified Checkpoint: {CHECKPOINT_PATH}")
    
    # 2. Select strictly untouched subset
    all_images = sorted(glob.glob(str(IMAGES_DIR / "*.jpg")))
    if len(all_images) != 548:
        raise ValueError(f"Expected 548 validation images, found {len(all_images)}")
    
    eval_images = all_images[100:548]
    if len(eval_images) != 448:
        raise ValueError(f"Expected 448 images in the held-out set, got {len(eval_images)}")
    print(f"Verified Evaluation Subset: {len(eval_images)} images [100:548]")
    
    # Write selected images list for audit
    audit_file = PROJECT_ROOT / "scratch" / "phase8b_image_list.txt"
    with open(audit_file, "w") as f:
        f.write("\n".join(eval_images))
    print(f"Image list recorded for auditing: {audit_file}")
    
    # Initialize Model & Components
    print("\nLoading models and allocating CUDA...")
    t_load_start = time.time()
    model = YOLO(str(CHECKPOINT_PATH))
    model.to('cuda')
    sahi = AdaptiveSAHI(default_slice_width=SAHI_SLICE, default_slice_height=SAHI_SLICE, default_overlap=SAHI_OVERLAP)
    t_load = time.time() - t_load_start
    print(f"Model initialization time: {t_load:.2f}s")
    
    # Warmup
    print("Performing CUDA warmup...")
    dummy = Image.new("RGB", (1536, 1536))
    for _ in range(3):
        _ = model(dummy, imgsz=IMGSZ, conf=CONF_THRESH, iou=NMS_IOU, verbose=False)
    torch.cuda.synchronize()
    
    # Evaluator
    evaluator = ComprehensiveEvaluator(val_dir=VAL_DIR, iou_threshold=0.50)
    
    base_preds = {}
    base_latencies = {}
    sahi_preds = {}
    sahi_latencies = {}
    
    total_base_time = 0
    total_sahi_time = 0
    
    print("\nStarting evaluation loop...")
    torch.cuda.reset_peak_memory_stats()
    
    for img_path in eval_images:
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        
        # CONDITION 1: Base E3
        torch.cuda.synchronize()
        t0 = time.time()
        res = model(img, imgsz=IMGSZ, conf=CONF_THRESH, iou=NMS_IOU, verbose=False)
        torch.cuda.synchronize()
        t_base = time.time() - t0
        
        base_preds[img_name] = extract_preds_from_ultralytics(res)
        base_latencies[img_name] = t_base
        total_base_time += t_base
        
        # CONDITION 2: E3 + SAHI
        def sahi_detect_fn(patches):
            batch_res = model(patches, imgsz=640, conf=CONF_THRESH, iou=NMS_IOU, verbose=False)
            batch_dets = []
            for r in batch_res:
                dets = []
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = r.names[cls_id]
                    dets.append(Detection(label=label, confidence=conf, bbox=[x1, y1, x2, y2]))
                batch_dets.append(dets)
            return batch_dets
            
        torch.cuda.synchronize()
        t0 = time.time()
        global_dets = sahi.run_sliced_inference(img, sahi_detect_fn, strategy="standard")
        merged = minimal_nms(global_dets, iou_threshold=SAHI_NMS)
        torch.cuda.synchronize()
        t_sahi = time.time() - t0
        
        # Convert Detection objects back to dicts for the evaluator
        sahi_preds[img_name] = [{"bbox": d.bbox, "confidence": d.confidence, "label": d.label} for d in merged]
        sahi_latencies[img_name] = t_sahi
        total_sahi_time += t_sahi
        
        print(f"Processed {img_name}: Base {t_base:.2f}s | SAHI {t_sahi:.2f}s")
        
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    print("\nComputing final COCO metrics...")
    base_results = evaluator.evaluate_predictions(base_preds, "E3 Baseline", base_latencies)
    sahi_results = evaluator.evaluate_predictions(sahi_preds, "E3 + SAHI", sahi_latencies)
    
    # Build final report
    out_report = {
        "metadata": {
            "images_evaluated": 448,
            "image_range": "[100:548]",
            "imgsz": IMGSZ,
            "conf": CONF_THRESH,
            "nms": NMS_IOU,
            "sahi_config": {"slice": SAHI_SLICE, "overlap": SAHI_OVERLAP, "nms": SAHI_NMS},
            "peak_vram_gb": peak_vram,
            "model_load_time_s": t_load
        },
        "condition_1_base": {
            "map50": base_results["overall"]["map50"],
            "map50_95": base_results["overall"]["map50_95"],
            "precision": base_results["overall"]["precision"],
            "recall": base_results["overall"]["recall"],
            "f1": base_results["overall"]["f1"],
            "tp": base_results["overall"].get("total_tp", "N/A"),
            "fp": base_results["overall"].get("total_fp", "N/A"),
            "fn": base_results["overall"].get("total_fn", "N/A"),
            "total_runtime_s": total_base_time,
            "mean_latency_ms": (total_base_time / 448) * 1000,
        },
        "condition_2_sahi": {
            "map50": sahi_results["overall"]["map50"],
            "map50_95": sahi_results["overall"]["map50_95"],
            "precision": sahi_results["overall"]["precision"],
            "recall": sahi_results["overall"]["recall"],
            "f1": sahi_results["overall"]["f1"],
            "tp": sahi_results["overall"].get("total_tp", "N/A"),
            "fp": sahi_results["overall"].get("total_fp", "N/A"),
            "fn": sahi_results["overall"].get("total_fn", "N/A"),
            "total_runtime_s": total_sahi_time,
            "mean_latency_ms": (total_sahi_time / 448) * 1000,
            "runtime_multiplier": total_sahi_time / total_base_time if total_base_time > 0 else 0,
            "images_processed_with_sahi": 448
        }
    }
    
    json_path = PROJECT_ROOT / "scratch" / "phase8b_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(out_report, f, indent=2)
        
    print(f"\nBenchmark Complete! Peak VRAM: {peak_vram:.2f} GB")
    print(json.dumps(out_report, indent=2))
    print(f"Results saved to {json_path}")

if __name__ == "__main__":
    run_benchmark()
