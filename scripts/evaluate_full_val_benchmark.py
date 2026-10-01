#!/usr/bin/env python3
"""
Full Validation Benchmark Evaluation across All 548 Images of VisDrone2019-DET-val.

Systems Evaluated:
1. Baseline YOLO-World (Zero-Shot, tuned conf=0.05)
2. SAHI Pipeline (Fine-Tuned, 640x640 slicing, 0.2 overlap, tuned conf=0.35)
3. AgentSearch-UAV Multi-Agent Pipeline (Fine-Tuned, adaptive SAHI, class-adaptive NMS, tuned conf=0.35)

Features:
- Checkpointing after each image: progress is never lost.
- Automatic resume from checkpoint.
- Microsecond latency profiling per image.
- Standardized evaluation with ComprehensiveEvaluator (IoU = 0.50).
- Generates publication-ready CSV and JSON reports.
"""

import os
import sys
import json
import csv
import time
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.postprocess import set_postprocess_backend
set_postprocess_backend("numpy")

# Setup paths
_current = Path(__file__).resolve().parent
PROJECT_ROOT = _current.parent.parent if _current.parent.name == "#FILLERS" else _current.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "#FILLERS"))

from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, CLASS_NAMES
from models.optimized_detection_pipeline import OptimizedDetectionPipeline


def get_checkpoint_path(output_dir: Path, system_key: str) -> Path:
    return output_dir / f"checkpoint_{system_key}_val548.json"


def get_latency_path(output_dir: Path, system_key: str) -> Path:
    return output_dir / f"latencies_{system_key}_val548.json"


def load_checkpoint(checkpoint_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint {checkpoint_path}: {e}")
            return {}
    return {}


def load_latencies(latency_path: Path) -> List[float]:
    if latency_path.exists():
        try:
            with open(latency_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load latencies {latency_path}: {e}")
            return []
    return []


def save_checkpoint(checkpoint_path: Path, data: Dict[str, List[Dict[str, Any]]]):
    temp_path = checkpoint_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    temp_path.replace(checkpoint_path)


def save_latencies(latency_path: Path, data: List[float]):
    temp_path = latency_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    temp_path.replace(latency_path)


def run_baseline_evaluation(eval_images: List[Path], output_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[float]]:
    system_key = "baseline"
    checkpoint_path = get_checkpoint_path(output_dir, system_key)
    latency_path = get_latency_path(output_dir, system_key)
    
    predictions = load_checkpoint(checkpoint_path)
    latencies = load_latencies(latency_path)
    
    weights_baseline = str(PROJECT_ROOT / "weights" / "yolov8s-world.pt")
    print(f"\n[1/3] Running Baseline YOLO-World on {len(eval_images)} images (conf=0.05)...")
    print(f"Loaded {len(predictions)} already completed predictions from checkpoint.")
    
    remaining_images = [p for p in eval_images if p.name not in predictions]
    if remaining_images:
        model = YOLO(weights_baseline)
        if hasattr(model, "set_classes"):
            model.set_classes(CLASS_NAMES)
            
        save_counter = 0
        for img_path in tqdm(remaining_images, desc="Baseline YOLO-World"):
            img_name = img_path.name
            img_str = str(img_path)
            
            t0 = time.perf_counter()
            res = model.predict(img_str, conf=0.05, verbose=False)
            t_elapsed = time.perf_counter() - t0
            latencies.append(t_elapsed)
            
            preds = []
            if res and len(res[0].boxes) > 0:
                for box in res[0].boxes:
                    cid = int(box.cls)
                    if 0 <= cid < len(CLASS_NAMES):
                        preds.append({
                            "class_id": cid,
                            "label": CLASS_NAMES[cid],
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist(),
                            "source": "baseline"
                        })
            predictions[img_name] = preds
            save_counter += 1
            if save_counter % 10 == 0:
                save_checkpoint(checkpoint_path, predictions)
                save_latencies(latency_path, latencies)
                
        save_checkpoint(checkpoint_path, predictions)
        save_latencies(latency_path, latencies)
        
    print(f"[1/3] Baseline YOLO-World completed: {len(predictions)} images processed.")
    return predictions, latencies


def run_sahi_evaluation(eval_images: List[Path], output_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[float]]:
    system_key = "sahi"
    checkpoint_path = get_checkpoint_path(output_dir, system_key)
    latency_path = get_latency_path(output_dir, system_key)
    
    predictions = load_checkpoint(checkpoint_path)
    latencies = load_latencies(latency_path)
    
    weights_best = str(PROJECT_ROOT / "runs" / "detect" / "experiments" / "model_search" / "E3_yolo11l_1536_aug" / "weights" / "best.pt")
    print(f"\n[2/3] Running SAHI Pipeline on {len(eval_images)} images (conf=0.35)...")
    print(f"Loaded {len(predictions)} already completed predictions from checkpoint.")
    
    remaining_images = [p for p in eval_images if p.name not in predictions]
    if remaining_images:
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=weights_best,
            confidence_threshold=0.35,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        save_counter = 0
        for img_path in tqdm(remaining_images, desc="SAHI Pipeline"):
            img_name = img_path.name
            img_str = str(img_path)
            
            t0 = time.perf_counter()
            sahi_res = get_sliced_prediction(
                img_str,
                sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0
            )
            t_elapsed = time.perf_counter() - t0
            latencies.append(t_elapsed)
            
            preds = []
            for obj in sahi_res.object_prediction_list:
                lbl = obj.category.name.lower()
                if lbl in CLASS_NAMES:
                    cid = CLASS_NAMES.index(lbl)
                else:
                    cid = int(obj.category.id)
                if 0 <= cid < len(CLASS_NAMES):
                    preds.append({
                        "class_id": cid,
                        "label": CLASS_NAMES[cid],
                        "confidence": float(obj.score.value),
                        "bbox": [
                            float(obj.bbox.minx),
                            float(obj.bbox.miny),
                            float(obj.bbox.maxx),
                            float(obj.bbox.maxy)
                        ],
                        "source": "sahi"
                    })
            predictions[img_name] = preds
            save_counter += 1
            if save_counter % 5 == 0:
                save_checkpoint(checkpoint_path, predictions)
                save_latencies(latency_path, latencies)
                
        save_checkpoint(checkpoint_path, predictions)
        save_latencies(latency_path, latencies)
        
    print(f"[2/3] SAHI Pipeline completed: {len(predictions)} images processed.")
    return predictions, latencies


def run_agentsearch_evaluation(eval_images: List[Path], output_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[float]]:
    system_key = "agentsearch"
    checkpoint_path = get_checkpoint_path(output_dir, system_key)
    latency_path = get_latency_path(output_dir, system_key)
    
    predictions = load_checkpoint(checkpoint_path)
    latencies = load_latencies(latency_path)
    
    weights_best = str(PROJECT_ROOT / "runs" / "detect" / "experiments" / "model_search" / "E3_yolo11l_1536_aug" / "weights" / "best.pt")
    config_det = str(PROJECT_ROOT / "configs" / "detection_config.json")
    
    print(f"\n[3/3] Running AgentSearch-UAV Multi-Agent Pipeline on {len(eval_images)} images (calibrated conf)...")
    print(f"Loaded {len(predictions)} already completed predictions from checkpoint.")
    
    remaining_images = [p for p in eval_images if p.name not in predictions]
    if remaining_images:
        pipeline = OptimizedDetectionPipeline(
            model_path=weights_best,
            config_path=config_det,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        save_counter = 0
        for img_path in tqdm(remaining_images, desc="AgentSearch-UAV"):
            img_name = img_path.name
            img_str = str(img_path)
            
            t0 = time.perf_counter()
            res = pipeline.detect_image(img_str, confidence_override=None)
            t_elapsed = time.perf_counter() - t0
            latencies.append(t_elapsed)
            
            preds = []
            for det in res.filtered_detections:
                lbl = det.label.lower()
                cid = CLASS_NAMES.index(lbl) if lbl in CLASS_NAMES else int(det.class_id if det.class_id is not None else 0)
                if 0 <= cid < len(CLASS_NAMES):
                    preds.append({
                        "class_id": cid,
                        "label": CLASS_NAMES[cid],
                        "confidence": float(det.confidence),
                        "bbox": det.bbox,
                        "source": det.source
                    })
            predictions[img_name] = preds
            save_counter += 1
            if save_counter % 5 == 0:
                save_checkpoint(checkpoint_path, predictions)
                save_latencies(latency_path, latencies)
                
        save_checkpoint(checkpoint_path, predictions)
        save_latencies(latency_path, latencies)
        
    print(f"[3/3] AgentSearch-UAV completed: {len(predictions)} images processed.")
    return predictions, latencies


def evaluate_and_generate_report(val_dir: Path, output_dir: Path):
    evaluator = ComprehensiveEvaluator(val_dir=val_dir, iou_threshold=0.50)
    system_names = {
        "baseline": "Baseline YOLO-World",
        "sahi": "YOLO-World + SAHI",
        "agentsearch": "AgentSearch-UAV (Multi-Agent Pipeline)"
    }
    
    summary_results = []
    full_reports = {}
    
    print("\n" + "="*80)
    print("CALCULATING OFFICIAL BENCHMARK METRICS ON 548 VISDRONE VAL IMAGES (IoU = 0.50)")
    print("="*80)
    
    for key, display_name in system_names.items():
        chk_path = get_checkpoint_path(output_dir, key)
        lat_path = get_latency_path(output_dir, key)
        
        if not chk_path.exists():
            print(f"[Warning] No predictions found for {display_name} at {chk_path}")
            continue
            
        preds = load_checkpoint(chk_path)
        latencies = load_latencies(lat_path)
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        
        metrics = evaluator.evaluate_predictions(preds, system_name=display_name)
        full_reports[display_name] = metrics
        
        row = {
            "System": display_name,
            "Precision": f"{metrics['overall']['precision'] * 100:.2f}%",
            "Recall": f"{metrics['overall']['recall'] * 100:.2f}%",
            "mAP@50": f"{metrics['overall']['map50'] * 100:.2f}%",
            "mAP@50-95": f"{metrics['overall']['map50_95'] * 100:.2f}%",
            "F1-Score": f"{metrics['overall']['f1'] * 100:.2f}%",
            "Total_Detections": sum(len(p) for p in preds.values()),
            "Total_TP": metrics['overall']['total_tp'],
            "Total_FP": metrics['overall']['total_fp'],
            "Total_FN": metrics['overall']['total_fn'],
            "Total_GT": metrics['overall']['total_gt'],
            "Avg_Latency_sec": round(avg_latency, 3),
            "Images_Evaluated": len(preds)
        }
        summary_results.append(row)
        
        print(f"\n--- {display_name} ---")
        print(f"  Precision:        {row['Precision']}")
        print(f"  Recall:           {row['Recall']}")
        print(f"  mAP@50:           {row['mAP@50']}")
        print(f"  mAP@50-95:        {row['mAP@50-95']}")
        print(f"  F1-Score:         {row['F1-Score']}")
        print(f"  TP / FP / FN:     {row['Total_TP']} / {row['Total_FP']} / {row['Total_FN']}")
        print(f"  Total Detections: {row['Total_Detections']}")
        print(f"  Avg Latency:      {row['Avg_Latency_sec']}s / image")
        print(f"  Images:           {row['Images_Evaluated']} / 548")
        
    # Save CSV
    csv_path = output_dir / "full_val_548_comparison.csv"
    if summary_results:
        fieldnames = list(summary_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_results)
        print(f"\nSaved CSV summary to: {csv_path}")
        
    # Save JSON
    json_path = output_dir / "full_val_548_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_results,
            "detailed_reports": full_reports
        }, f, indent=2)
    print(f"Saved JSON reports to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Full 548-image VisDrone2019-DET-val benchmark.")
    parser.add_argument("--system", choices=["all", "baseline", "sahi", "agentsearch"], default="all",
                        help="Which system(s) to evaluate.")
    parser.add_argument("--eval-only", action="store_true", help="Only calculate metrics from existing checkpoints.")
    args = parser.parse_args()
    
    candidates = [
        PROJECT_ROOT / "#FILLERS" / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
        PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
    ]
    val_dir = next((c for c in candidates if c.exists()), candidates[0])
    images_dir = val_dir / "images"
    output_dir = PROJECT_ROOT / "#FILLERS" / "experiments" / "all_systems_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    print(f"Found {len(all_images)} total validation images in {images_dir}")
    
    if not args.eval_only:
        if args.system in ["all", "baseline"]:
            run_baseline_evaluation(all_images, output_dir)
        if args.system in ["all", "sahi"]:
            run_sahi_evaluation(all_images, output_dir)
        if args.system in ["all", "agentsearch"]:
            run_agentsearch_evaluation(all_images, output_dir)
            
    evaluate_and_generate_report(val_dir, output_dir)


if __name__ == "__main__":
    main()
