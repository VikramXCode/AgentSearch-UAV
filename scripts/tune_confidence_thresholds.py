#!/usr/bin/env python3
"""
Confidence Threshold Tuning Script for AgentSearch-UAV Benchmarking.

Evaluates multiple confidence thresholds across the 3 evaluated systems:
1. Baseline YOLO-World (Zero-Shot)
2. YOLO-World + SAHI (Sliced Inference)
3. Full AgentSearch-UAV Multi-Agent System

Finds the optimal threshold maximizing F1-score while maintaining
a balanced Precision/Recall trade-off on the 30 VisDrone2019-DET-val images.
No change to model architectures, weights, datasets, or evaluation methodology.
"""

from __future__ import annotations

import os
import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
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
from utils.class_threshold_calibration import DEFAULT_CLASS_THRESHOLDS


def select_best_balanced_threshold(
    results: List[Dict[str, Any]], 
    f1_tolerance: float = 0.005,
    min_score: float = 0.05,
) -> Dict[str, Any]:
    """
    Selects the threshold that achieves peak F1-score while maintaining
    a reasonable balance between Precision and Recall.
    
    If multiple candidate thresholds achieve F1 within `f1_tolerance` of the
    maximum F1, the candidate with the smallest |Precision - Recall| gap is chosen.
    """
    valid = [r for r in results if r.get("precision", 0) >= min_score and r.get("recall", 0) >= min_score]
    if not valid:
        valid = results

    max_f1 = max(r["f1"] for r in valid)
    top_contenders = [r for r in valid if r["f1"] >= max_f1 - f1_tolerance]
    
    # Minimize |Precision - Recall|, maximize F1 as secondary tie-breaker
    best = min(top_contenders, key=lambda r: (abs(r["precision"] - r["recall"]), -r["f1"]))
    return best


def tune_thresholds_for_systems(num_images: int = 30):
    candidates = [
        PROJECT_ROOT / "#FILLERS" / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
        PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
    ]
    val_dir = next((c for c in candidates if c.exists()), candidates[0])
    images_dir = val_dir / "images"
    evaluator = ComprehensiveEvaluator(val_dir=val_dir, iou_threshold=0.50)

    all_image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    eval_images = all_image_paths[:num_images] if num_images > 0 else all_image_paths

    total_gt = sum(len(evaluator.ground_truths.get(p.name, [])) for p in eval_images)

    print(f"\n=================================================================")
    print(f"CONFIDENCE THRESHOLD TUNING ON {len(eval_images)} VISDRONE VAL IMAGES")
    print(f"Total Ground-Truth Objects: {total_gt}")
    print(f"=================================================================\n")

    candidate_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50]

    # -------------------------------------------------------------------------
    # 1. SYSTEM 1: BASELINE YOLO-WORLD
    # -------------------------------------------------------------------------
    print("\n--- System 1: Baseline YOLO-World ---")
    weights_baseline = str(PROJECT_ROOT / "weights" / "yolov8s-world.pt")
    m_baseline = YOLO(weights_baseline)
    if hasattr(m_baseline, "set_classes"):
        m_baseline.set_classes(CLASS_NAMES)

    raw_baseline_preds: Dict[str, List[Dict[str, Any]]] = {}
    print("Collecting raw predictions for Baseline YOLO-World (conf=0.02)...")
    for img_path in tqdm(eval_images, desc="Baseline inference"):
        img_name = img_path.name
        res = m_baseline.predict(str(img_path), conf=0.02, verbose=False)
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
        raw_baseline_preds[img_name] = preds

    baseline_tuning_results = []
    for conf in candidate_thresholds:
        filtered = {}
        for img_name, p_list in raw_baseline_preds.items():
            filtered[img_name] = [p for p in p_list if p["confidence"] >= conf]
        
        eval_metrics = evaluator.evaluate_predictions(filtered, system_name=f"Baseline (conf={conf:.2f})")
        ov = eval_metrics["overall"]
        baseline_tuning_results.append({
            "conf": conf,
            "precision": ov["precision"],
            "recall": ov["recall"],
            "f1": ov["f1"],
            "map50": ov["map50"],
            "total_tp": ov["total_tp"],
            "total_fp": ov["total_fp"],
            "total_fn": ov["total_fn"],
            "total_preds": sum(len(p) for p in filtered.values()),
            "precision_recall_gap": abs(ov["precision"] - ov["recall"])
        })

    best_baseline = select_best_balanced_threshold(baseline_tuning_results, f1_tolerance=0.005)
    print(f"\nBaseline Tuning Results:")
    for r in baseline_tuning_results:
        flag = " <--- SELECTED OPTIMAL (Best Balanced F1)" if r["conf"] == best_baseline["conf"] else ""
        print(f"  Conf: {r['conf']:.2f} | Precision: {r['precision']*100:.2f}% | Recall: {r['recall']*100:.2f}% | F1: {r['f1']*100:.2f}% | mAP@50: {r['map50']*100:.2f}% | TP: {r['total_tp']} | FP: {r['total_fp']}{flag}")

    # -------------------------------------------------------------------------
    # 2. SYSTEM 2: YOLO-WORLD + SAHI
    # -------------------------------------------------------------------------
    print("\n--- System 2: YOLO-World + SAHI ---")
    weights_best = str(PROJECT_ROOT / "weights" / "best.pt")
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights_best,
        confidence_threshold=0.05,
        device="cpu"
    )

    raw_sahi_preds: Dict[str, List[Dict[str, Any]]] = {}
    print("Collecting raw predictions for YOLO-World + SAHI (conf=0.05)...")
    for img_path in tqdm(eval_images, desc="SAHI inference"):
        img_name = img_path.name
        sahi_res = get_sliced_prediction(
            str(img_path),
            sahi_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0
        )
        preds = []
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
                preds.append({
                    "class_id": cid,
                    "label": CLASS_NAMES[cid],
                    "confidence": float(obj.score.value),
                    "bbox": bbox,
                    "source": "sahi"
                })
        raw_sahi_preds[img_name] = preds

    sahi_tuning_results = []
    for conf in candidate_thresholds:
        filtered = {}
        for img_name, p_list in raw_sahi_preds.items():
            filtered[img_name] = [p for p in p_list if p["confidence"] >= conf]
        
        eval_metrics = evaluator.evaluate_predictions(filtered, system_name=f"SAHI (conf={conf:.2f})")
        ov = eval_metrics["overall"]
        sahi_tuning_results.append({
            "conf": conf,
            "precision": ov["precision"],
            "recall": ov["recall"],
            "f1": ov["f1"],
            "map50": ov["map50"],
            "total_tp": ov["total_tp"],
            "total_fp": ov["total_fp"],
            "total_fn": ov["total_fn"],
            "total_preds": sum(len(p) for p in filtered.values()),
            "precision_recall_gap": abs(ov["precision"] - ov["recall"])
        })

    best_sahi = select_best_balanced_threshold(sahi_tuning_results, f1_tolerance=0.005)
    print(f"\nSAHI Tuning Results:")
    for r in sahi_tuning_results:
        flag = " <--- SELECTED OPTIMAL (Best Balanced F1)" if r["conf"] == best_sahi["conf"] else ""
        print(f"  Conf: {r['conf']:.2f} | Precision: {r['precision']*100:.2f}% | Recall: {r['recall']*100:.2f}% | F1: {r['f1']*100:.2f}% | mAP@50: {r['map50']*100:.2f}% | TP: {r['total_tp']} | FP: {r['total_fp']}{flag}")

    # -------------------------------------------------------------------------
    # 3. SYSTEM 3: FULL AGENTSEARCH-UAV MULTI-AGENT SYSTEM
    # -------------------------------------------------------------------------
    print("\n--- System 3: Full AgentSearch-UAV Multi-Agent System ---")
    config_det = str(PROJECT_ROOT / "configs" / "detection_config.json")
    multi_agent_pipeline = OptimizedDetectionPipeline(
        model_path=weights_best,
        config_path=config_det
    )

    # Load baseline calibrated thresholds
    class_thresh_file = PROJECT_ROOT / "configs" / "class_thresholds.json"
    if class_thresh_file.exists():
        with open(class_thresh_file, "r", encoding="utf-8") as f:
            saved_thresh = json.load(f).get("thresholds", DEFAULT_CLASS_THRESHOLDS)
    else:
        saved_thresh = DEFAULT_CLASS_THRESHOLDS

    # Collect unclipped detections from AgentSearch-UAV pipeline
    raw_agent_preds: Dict[str, List[Dict[str, Any]]] = {}
    print("Collecting detections for Full AgentSearch-UAV Multi-Agent System (unclipped)...")
    for img_path in tqdm(eval_images, desc="AgentSearch-UAV inference"):
        img_name = img_path.name
        res = multi_agent_pipeline.detect_image(str(img_path), confidence_override="none")
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
                    "source": det.source,
                    "scale_category": getattr(det, "scale_category", "medium")
                })
        raw_agent_preds[img_name] = preds

    # Sweep both calibrated multipliers and uniform thresholds
    agent_sweep_configs = []
    
    # 1. Calibrated scale multipliers
    multipliers = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
    for mult in multipliers:
        agent_sweep_configs.append({
            "type": "calibrated_multiplier",
            "multiplier": mult,
            "label": f"calibrated x {mult:.2f}",
            "conf_val": round(mult, 2)
        })

    # 2. Uniform global thresholds
    uniform_thresholds = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]
    for ut in uniform_thresholds:
        agent_sweep_configs.append({
            "type": "uniform",
            "threshold": ut,
            "label": f"uniform (conf={ut:.2f})",
            "conf_val": ut
        })

    agent_tuning_results = []
    for cfg in agent_sweep_configs:
        filtered = {}
        for img_name, p_list in raw_agent_preds.items():
            img_filtered = []
            for p in p_list:
                lbl = p["label"]
                conf = p["confidence"]
                scale = p.get("scale_category", "medium")

                if cfg["type"] == "uniform":
                    keep = conf >= cfg["threshold"]
                else:
                    mult = cfg["multiplier"]
                    base_t = saved_thresh.get(lbl, 0.25) * mult
                    if scale == "tiny":
                        eff_t = max(0.08, base_t - 0.05)
                    elif scale == "small":
                        eff_t = max(0.10, base_t - 0.025)
                    elif scale == "large":
                        eff_t = min(0.60, base_t + 0.05)
                    else:
                        eff_t = base_t
                    keep = conf >= eff_t

                if keep:
                    img_filtered.append(p)
            filtered[img_name] = img_filtered

        eval_metrics = evaluator.evaluate_predictions(filtered, system_name=f"AgentSearch-UAV ({cfg['label']})")
        ov = eval_metrics["overall"]
        agent_tuning_results.append({
            "config": cfg,
            "conf": cfg["conf_val"],
            "label": cfg["label"],
            "precision": ov["precision"],
            "recall": ov["recall"],
            "f1": ov["f1"],
            "map50": ov["map50"],
            "total_tp": ov["total_tp"],
            "total_fp": ov["total_fp"],
            "total_fn": ov["total_fn"],
            "total_preds": sum(len(p) for p in filtered.values()),
            "precision_recall_gap": abs(ov["precision"] - ov["recall"])
        })

    best_agent = select_best_balanced_threshold(agent_tuning_results, f1_tolerance=0.005)
    print(f"\nAgentSearch-UAV Tuning Results:")
    for r in agent_tuning_results:
        flag = " <--- SELECTED OPTIMAL (Best Balanced F1)" if r["label"] == best_agent["label"] else ""
        print(f"  {r['label']:<24} | Precision: {r['precision']*100:.2f}% | Recall: {r['recall']*100:.2f}% | F1: {r['f1']*100:.2f}% | mAP@50: {r['map50']*100:.2f}% | TP: {r['total_tp']} | FP: {r['total_fp']}{flag}")

    # -------------------------------------------------------------------------
    # 4. SAVE TUNING SUMMARY JSON
    # -------------------------------------------------------------------------
    out_dir = PROJECT_ROOT / "#FILLERS" / "experiments" / "all_systems_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    tuning_summary_path = out_dir / "confidence_tuning_results.json"

    tuning_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "VisDrone2019-DET-val (30 images, 1430 GT)",
        "optimal_thresholds": {
            "baseline": {
                "system": "Baseline YOLO-World (Zero-Shot)",
                "optimal_threshold": best_baseline["conf"],
                "precision": round(best_baseline["precision"], 4),
                "recall": round(best_baseline["recall"], 4),
                "f1": round(best_baseline["f1"], 4),
                "map50": round(best_baseline["map50"], 4),
                "tp": best_baseline["total_tp"],
                "fp": best_baseline["total_fp"],
                "fn": best_baseline["total_fn"],
            },
            "sahi": {
                "system": "YOLO-World + SAHI (Sliced Inference)",
                "optimal_threshold": best_sahi["conf"],
                "precision": round(best_sahi["precision"], 4),
                "recall": round(best_sahi["recall"], 4),
                "f1": round(best_sahi["f1"], 4),
                "map50": round(best_sahi["map50"], 4),
                "tp": best_sahi["total_tp"],
                "fp": best_sahi["total_fp"],
                "fn": best_sahi["total_fn"],
            },
            "agentsearch": {
                "system": "AgentSearch-UAV (Multi-Agent)",
                "optimal_configuration": best_agent["label"],
                "optimal_threshold": best_agent["conf"],
                "config_detail": best_agent["config"],
                "precision": round(best_agent["precision"], 4),
                "recall": round(best_agent["recall"], 4),
                "f1": round(best_agent["f1"], 4),
                "map50": round(best_agent["map50"], 4),
                "tp": best_agent["total_tp"],
                "fp": best_agent["total_fp"],
                "fn": best_agent["total_fn"],
            }
        },
        "sweeps": {
            "baseline": baseline_tuning_results,
            "sahi": sahi_tuning_results,
            "agentsearch": agent_tuning_results
        }
    }

    with open(tuning_summary_path, "w", encoding="utf-8") as f:
        json.dump(tuning_data, f, indent=2)
    print(f"\nTuning results successfully saved to: {tuning_summary_path}")

    return tuning_data


if __name__ == "__main__":
    tune_thresholds_for_systems(num_images=30)
