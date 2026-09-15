#!/usr/bin/env python3
"""
Comprehensive Benchmark Evaluation across All 5 Detection Systems on VisDrone2019-DET-val:
1. Baseline YOLO-World
2. Fine-Tuned YOLO-World
3. SAHI Pipeline
4. Super-Resolution Pipeline
5. AgentSearch-UAV Multi-Agent Pipeline

Evaluated using identical validation ground truth, class mapping, IoU thresholds, and evaluation logic.
No hardcoded or generated metrics.
"""

import os
import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

_current = Path(__file__).resolve().parent
PROJECT_ROOT = _current.parent.parent if _current.parent.name == "#FILLERS" else _current.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "#FILLERS"))

from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, CLASS_NAMES
from models.optimized_detection_pipeline import OptimizedDetectionPipeline
from models.super_resolution import SuperResolutionEngine


def run_all_systems_benchmark(num_images: int = 100):
    candidates = [
        PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
        PROJECT_ROOT / "#FILLERS" / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
    ]
    val_dir = next((c for c in candidates if c.exists()), candidates[0])
    images_dir = val_dir / "images"
    output_dir = PROJECT_ROOT / "experiments" / "all_systems_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ComprehensiveEvaluator(val_dir=val_dir, iou_threshold=0.50)
    all_image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    if num_images > 0:
        eval_images = all_image_paths[:num_images]
    else:
        eval_images = all_image_paths

    print(f"\n=================================================================")
    print(f"BENCHMARKING 5 DETECTION SYSTEMS ON {len(eval_images)} VISDRONE VAL IMAGES")
    print(f"=================================================================\n")

    # 1. Initialize all models
    print("Loading models...")
    
    weights_baseline = str(PROJECT_ROOT / "weights" / "yolov8s-world.pt")
    weights_best = str(PROJECT_ROOT / "weights" / "best.pt")
    config_det = str(PROJECT_ROOT / "configs" / "detection_config.json")

    # System 1: Baseline YOLO-World (tuned optimal conf=0.05)
    m_baseline = YOLO(weights_baseline)
    if hasattr(m_baseline, "set_classes"):
        m_baseline.set_classes(CLASS_NAMES)
        
    # System 2: Fine-Tuned YOLO-World
    m_finetuned = YOLO(weights_best)
    
    # System 3: SAHI Pipeline (tuned optimal conf=0.35)
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights_best,
        confidence_threshold=0.35,
        device="cpu"
    )
    
    # System 4: Super-Resolution Pipeline
    sr_engine = SuperResolutionEngine()
    
    # System 5: AgentSearch-UAV Multi-Agent Pipeline (tuned optimal conf=0.35)
    multi_agent_pipeline = OptimizedDetectionPipeline(
        model_path=weights_best,
        config_path=config_det
    )

    systems = [
        "Baseline YOLO-World",
        "Fine-Tuned YOLO-World",
        "SAHI Pipeline",
        "Super-Resolution Pipeline",
        "AgentSearch-UAV Multi-Agent Pipeline"
    ]

    all_predictions: Dict[str, Dict[str, List[Dict[str, Any]]]] = {sys_name: {} for sys_name in systems}
    system_latencies: Dict[str, List[float]] = {sys_name: [] for sys_name in systems}

    # Evaluate across images
    for img_path in tqdm(eval_images, desc="Evaluating Images across Systems"):
        img_name = img_path.name
        img_str = str(img_path)

        # ----------------------------------------------------
        # 1. Baseline YOLO-World (tuned optimal conf=0.05)
        # ----------------------------------------------------
        t0 = time.perf_counter()
        res1 = m_baseline.predict(img_str, conf=0.05, verbose=False)
        t_base = time.perf_counter() - t0
        system_latencies["Baseline YOLO-World"].append(t_base)

        preds1 = []
        if res1 and len(res1[0].boxes) > 0:
            for box in res1[0].boxes:
                cid = int(box.cls)
                if 0 <= cid < len(CLASS_NAMES):
                    preds1.append({
                        "class_id": cid,
                        "label": CLASS_NAMES[cid],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                        "source": "baseline"
                    })
        all_predictions["Baseline YOLO-World"][img_name] = preds1

        # ----------------------------------------------------
        # 2. Fine-Tuned YOLO-World
        # ----------------------------------------------------
        t0 = time.perf_counter()
        res2 = m_finetuned.predict(img_str, conf=0.25, verbose=False)
        t_ft = time.perf_counter() - t0
        system_latencies["Fine-Tuned YOLO-World"].append(t_ft)

        preds2 = []
        if res2 and len(res2[0].boxes) > 0:
            for box in res2[0].boxes:
                cid = int(box.cls)
                if 0 <= cid < len(CLASS_NAMES):
                    preds2.append({
                        "class_id": cid,
                        "label": CLASS_NAMES[cid],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                        "source": "finetuned"
                    })
        all_predictions["Fine-Tuned YOLO-World"][img_name] = preds2

        # ----------------------------------------------------
        # 3. SAHI Pipeline
        # ----------------------------------------------------
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
        t_sahi = time.perf_counter() - t0
        system_latencies["SAHI Pipeline"].append(t_sahi)

        preds3 = []
        for obj in sahi_res.object_prediction_list:
            lbl = obj.category.name.lower()
            if lbl in CLASS_NAMES:
                cid = CLASS_NAMES.index(lbl)
            else:
                cid = int(obj.category.id)
            if 0 <= cid < len(CLASS_NAMES):
                bbox = [
                    float(obj.bbox.minx),
                    float(obj.bbox.miny),
                    float(obj.bbox.maxx),
                    float(obj.bbox.maxy)
                ]
                preds3.append({
                    "class_id": cid,
                    "label": CLASS_NAMES[cid],
                    "confidence": float(obj.score.value),
                    "bbox": bbox,
                    "source": "sahi"
                })
        all_predictions["SAHI Pipeline"][img_name] = preds3

        # ----------------------------------------------------
        # 4. Super-Resolution Pipeline
        # ----------------------------------------------------
        t0 = time.perf_counter()
        upscaled_img = sr_engine.upscale(img_str, scale=2)
        res4 = m_finetuned.predict(upscaled_img, conf=0.25, verbose=False)
        t_sr = time.perf_counter() - t0
        system_latencies["Super-Resolution Pipeline"].append(t_sr)

        preds4 = []
        if res4 and len(res4[0].boxes) > 0:
            for box in res4[0].boxes:
                cid = int(box.cls)
                if 0 <= cid < len(CLASS_NAMES):
                    orig_bbox = [float(c) * 0.5 for c in box.xyxy[0].tolist()]
                    preds4.append({
                        "class_id": cid,
                        "label": CLASS_NAMES[cid],
                        "confidence": float(box.conf),
                        "bbox": orig_bbox,
                        "source": "super_resolution"
                    })
        all_predictions["Super-Resolution Pipeline"][img_name] = preds4

        # ----------------------------------------------------
        # 5. AgentSearch-UAV Multi-Agent Pipeline (tuned optimal conf=0.35)
        # ----------------------------------------------------
        t0 = time.perf_counter()
        res5 = multi_agent_pipeline.detect_image(img_str, confidence_override=0.35)
        t_ma = time.perf_counter() - t0
        system_latencies["AgentSearch-UAV Multi-Agent Pipeline"].append(t_ma)

        preds5 = []
        for det in res5.filtered_detections:
            lbl = det.label.lower()
            cid = CLASS_NAMES.index(lbl) if lbl in CLASS_NAMES else int(det.class_id if det.class_id is not None else 0)
            if 0 <= cid < len(CLASS_NAMES):
                preds5.append({
                    "class_id": cid,
                    "label": CLASS_NAMES[cid],
                    "confidence": float(det.confidence),
                    "bbox": det.bbox,
                    "source": det.source
                })
        all_predictions["AgentSearch-UAV Multi-Agent Pipeline"][img_name] = preds5

    # --------------------------------------------------------
    # Calculate Standardized Metrics for Each System
    # --------------------------------------------------------
    summary_results = []
    full_eval_reports = {}

    print("\n=================================================================")
    print("CALCULATED BENCHMARK METRICS (IoU = 0.50, Same Ground Truth)")
    print("=================================================================")

    for sys_name in systems:
        preds = all_predictions[sys_name]
        eval_metrics = evaluator.evaluate_predictions(preds, system_name=sys_name)
        full_eval_reports[sys_name] = eval_metrics
        
        overall = eval_metrics["overall"]
        prec = overall["precision"]
        rec = overall["recall"]
        f1 = overall["f1"]
        map50 = overall["map50"]
        map50_95 = overall.get("map50_95", 0.0)
        avg_lat = float(np.mean(system_latencies[sys_name]))

        row = {
            "System": sys_name,
            "Precision": f"{prec * 100:.2f}%",
            "Recall": f"{rec * 100:.2f}%",
            "mAP@50": f"{map50 * 100:.2f}%",
            "F1-Score": f"{f1 * 100:.2f}%",
            "Precision_raw": prec,
            "Recall_raw": rec,
            "mAP50_raw": map50,
            "F1_raw": f1,
            "Avg_Latency_sec": round(avg_lat, 3),
            "Total_Detections": sum(len(p) for p in preds.values()),
            "Total_TP": overall["total_tp"],
            "Total_FP": overall["total_fp"],
            "Total_FN": overall["total_fn"],
            "Total_GT": overall["total_gt"]
        }
        summary_results.append(row)

    # Save summary to CSV
    csv_path = output_dir / "all_systems_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "System", "Precision", "Recall", "mAP@50", "F1-Score",
            "Precision_raw", "Recall_raw", "mAP50_raw", "F1_raw", "Avg_Latency_sec",
            "Total_Detections", "Total_TP", "Total_FP", "Total_FN", "Total_GT"
        ])
        writer.writeheader()
        writer.writerows(summary_results)

    # Save summary & full evaluation details to JSON
    json_path = output_dir / "all_systems_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_results,
            "detailed_reports": full_eval_reports
        }, f, indent=2)

    # Print Table for All Systems
    print("\n| System | Precision | Recall | mAP@50 | F1-Score |")
    print("|---|---|---|---|---|")
    for r in summary_results:
        print(f"| {r['System']} | {r['Precision']} | {r['Recall']} | {r['mAP@50']} | {r['F1-Score']} |")

    # Print Dedicated Table for the 3 Evaluated Systems
    print("\n=================================================================")
    print("THREE EVALUATED SYSTEMS BENCHMARK (TUNED OPTIMAL OPERATING POINTS)")
    print("=================================================================")
    print("| System | Precision | Recall | mAP@50 | F1-Score | TP | FP | FN |")
    print("|---|---|---|---|---|---|---|---|")
    for r in summary_results:
        if r['System'] in ["Baseline YOLO-World", "SAHI Pipeline", "AgentSearch-UAV Multi-Agent Pipeline"]:
            print(f"| {r['System']} | {r['Precision']} | {r['Recall']} | {r['mAP@50']} | {r['F1-Score']} | {r['Total_TP']} | {r['Total_FP']} | {r['Total_FN']} |")

    print(f"\nBenchmark completed successfully! Reports saved to {output_dir}")
    return summary_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark detection systems on VisDrone-val")
    parser.add_argument("pos_num_images", nargs="?", type=int, default=None, help="Number of images to evaluate (positional)")
    parser.add_argument("--num-images", "-n", type=int, default=30, help="Number of images to evaluate (default: 30)")
    args = parser.parse_args()

    n_imgs = args.pos_num_images if args.pos_num_images is not None else args.num_images
    run_all_systems_benchmark(num_images=n_imgs)
