#!/usr/bin/env python3
"""
Standardized Evaluation Script for the Optimized AgentSearch-UAV Pipeline (v2.5.0)
Evaluates PR-curves, scale breakdown, and comparative metrics against the baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, CLASS_NAMES
from models.optimized_detection_pipeline import OptimizedDetectionPipeline
from ultralytics import YOLO


def run_evaluation(num_images: int = 25, iou_thresh: float = 0.50):
    val_dir = PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val"
    images_dir = val_dir / "images"
    output_dir = PROJECT_ROOT / "experiments" / "optimized_pipeline_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ComprehensiveEvaluator(val_dir=val_dir, iou_threshold=iou_thresh)
    all_images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    if num_images > 0:
        eval_images = all_images[:num_images]
    else:
        eval_images = all_images

    print(f"\n=======================================================")
    print(f"EVALUATING OPTIMIZED PIPELINE ON {len(eval_images)} IMAGES")
    print(f"=======================================================\n")

    pipeline = OptimizedDetectionPipeline(
        model_path="weights/best.pt",
        config_path="configs/detection_config.json"
    )

    preds_dict = {}
    latencies = {}

    for img_p in tqdm(eval_images, desc="Running Optimized Pipeline"):
        t0 = time.perf_counter()
        res = pipeline.detect_image(str(img_p))
        elapsed = time.perf_counter() - t0
        latencies[img_p.name] = elapsed

        det_list = []
        for det in res.filtered_detections:
            lbl = det.label.lower()
            cid = CLASS_NAMES.index(lbl) if lbl in CLASS_NAMES else (det.class_id if det.class_id is not None else 0)
            det_list.append({
                "class_id": cid,
                "label": CLASS_NAMES[cid],
                "confidence": float(det.confidence),
                "bbox": det.bbox,
                "source": det.source,
                "scale_category": det.scale_category
            })
        preds_dict[img_p.name] = det_list

    results = evaluator.evaluate_predictions(
        predictions=preds_dict,
        system_name="AgentSearch-UAV Optimized Pipeline v2.5.0",
        latencies=latencies
    )

    out_file = output_dir / "evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n--- RESULTS SUMMARY ---")
    print(f"Precision : {results['overall']['precision']*100:.2f}%")
    print(f"Recall    : {results['overall']['recall']*100:.2f}%")
    print(f"F1-Score  : {results['overall']['f1']*100:.2f}%")
    print(f"mAP@50    : {results['overall']['map50']*100:.2f}%")
    print(f"mAP@50-95 : {results['overall']['map50_95']*100:.2f}%")
    print(f"\nSaved report to: {out_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=25)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()
    run_evaluation(num_images=args.num_images, iou_thresh=args.iou)
