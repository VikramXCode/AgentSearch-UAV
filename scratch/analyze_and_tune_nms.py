#!/usr/bin/env python3
"""
Analysis & Optimization Script for AgentUAV NMS and Post-Processing.

Tests multiple NMS algorithms (Standard, Linear Soft-NMS, Gaussian Soft-NMS, Hybrid Soft-NMS),
IoU thresholds, scale-aware maps, and confidence filtering to maximize F1, Precision,
and mAP@50 while reducing duplicate bounding boxes and false positives on the 30 VisDrone val images.
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from tqdm import tqdm

_current = Path(__file__).resolve().parent
PROJECT_ROOT = _current.parent.parent if _current.parent.name == "#FILLERS" else _current.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "#FILLERS"))

from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, CLASS_NAMES
from models.optimized_detection_pipeline import OptimizedDetectionPipeline
from models.schemas import Detection
from utils.improved_soft_nms import compute_iou, get_scale_category
from utils.class_conflict_resolver import ClassConflictResolver
from utils.false_positive_filter import FalsePositiveFilter


def run_nms_variant(
    items: List[Dict[str, Any]],
    method: str = "hybrid",
    iou_thresh: float = 0.35,
    duplicate_cutoff_iou: float = 0.55,
    score_thresh: float = 0.30,
    sigma: float = 0.35,
    score_decay: float = 0.85,
    use_scale_aware: bool = True,
    scale_iou_map: Dict[str, float] = None,
) -> List[Dict[str, Any]]:
    """
    Flexible NMS engine supporting:
    - standard (hard greedy)
    - linear (Soft-NMS)
    - gaussian (Soft-NMS)
    - hybrid (hard suppress if IoU >= duplicate_cutoff_iou, soft decay otherwise)
    """
    if len(items) <= 1:
        return [it for it in items if it["score"] >= score_thresh]

    # Sort descending by score
    pool = sorted(items, key=lambda x: x["score"], reverse=True)
    kept = []

    if scale_iou_map is None:
        scale_iou_map = {"tiny": 0.28, "small": 0.32, "medium": 0.38, "large": 0.45}

    while pool:
        best = pool.pop(0)
        if best["score"] < score_thresh:
            break
        kept.append(best)

        best_bbox = best["bbox"]
        if use_scale_aware:
            scale = get_scale_category(best_bbox)
            eff_iou = scale_iou_map.get(scale, iou_thresh)
        else:
            eff_iou = iou_thresh

        remaining = []
        for it in pool:
            iou = compute_iou(best_bbox, it["bbox"])

            if method == "standard":
                if iou < eff_iou:
                    remaining.append(it)
            elif method == "linear":
                if iou >= eff_iou:
                    it["score"] *= (1.0 - iou * score_decay)
                if it["score"] >= score_thresh:
                    remaining.append(it)
            elif method == "gaussian":
                if iou >= eff_iou:
                    weight = math.exp(-(iou * iou) / sigma)
                    it["score"] *= weight
                if it["score"] >= score_thresh:
                    remaining.append(it)
            elif method == "hybrid":
                # If IoU is very high, it is an unambiguous duplicate box from overlapping tiles
                if iou >= duplicate_cutoff_iou:
                    # Hard suppress duplicate
                    continue
                elif iou >= eff_iou:
                    # Soft continuous decay for adjacent / touching true objects
                    weight = math.exp(-(iou * iou) / sigma)
                    it["score"] *= weight
                if it["score"] >= score_thresh:
                    remaining.append(it)
            else:
                if iou < eff_iou:
                    remaining.append(it)

        pool = remaining
        if method in ("linear", "gaussian", "hybrid"):
            pool.sort(key=lambda x: x["score"], reverse=True)

    return kept


def main():
    candidates = [
        PROJECT_ROOT / "#FILLERS" / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
        PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
    ]
    val_dir = next((c for c in candidates if c.exists()), candidates[0])
    images_dir = val_dir / "images"
    evaluator = ComprehensiveEvaluator(val_dir=val_dir, iou_threshold=0.50)

    all_image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))[:30]

    cache_file = PROJECT_ROOT / "#FILLERS" / "scratch" / "raw_fused_detections_cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    raw_fused_by_img = {}
    if cache_file.exists():
        print(f"Loading cached pre-NMS fused detections from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            raw_fused_by_img = json.load(f)
    else:
        print("Collecting pre-NMS fused detections for 30 images...")
        pipeline = OptimizedDetectionPipeline(
            model_path=str(PROJECT_ROOT / "weights" / "best.pt"),
            config_path=str(PROJECT_ROOT / "configs" / "detection_config.json")
        )
        for img_p in tqdm(all_image_paths, desc="Pipeline inference (one-time)"):
            res = pipeline.detect_image(str(img_p), confidence_override="none")
            # Store detections
            dets_json = []
            for d in res.filtered_detections:
                dets_json.append({
                    "label": d.label.lower(),
                    "class_id": d.class_id,
                    "confidence": float(d.confidence),
                    "bbox": d.bbox,
                    "source": d.source,
                })
            raw_fused_by_img[img_p.name] = dets_json

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(raw_fused_by_img, f)
        print(f"Cached pre-NMS fused detections to {cache_file}")

    print(f"\nAnalyzing NMS configurations across {len(all_image_paths)} images (1430 GT targets)...")

    # Baseline configuration (before optimization)
    # Linear Soft-NMS, iou=0.35, conf=0.35
    print("\nEvaluating Baseline (Before Optimization)...")
    baseline_preds = {}
    for img_name, d_list in raw_fused_by_img.items():
        # group by class
        by_class = {}
        for d in d_list:
            by_class.setdefault(d["label"], []).append({
                "label": d["label"],
                "class_id": d["class_id"],
                "bbox": d["bbox"],
                "score": d["confidence"]
            })
        kept = []
        for lbl, group in by_class.items():
            kept.extend(run_nms_variant(
                group,
                method="linear",
                iou_thresh=0.35,
                score_thresh=0.35,
                score_decay=0.75,
                use_scale_aware=True,
                scale_iou_map={"tiny": 0.28, "small": 0.35, "medium": 0.40, "large": 0.50}
            ))
        baseline_preds[img_name] = [
            {"class_id": CLASS_NAMES.index(k["label"]), "label": k["label"], "confidence": k["score"], "bbox": k["bbox"]}
            for k in kept if k["label"] in CLASS_NAMES
        ]

    base_eval = evaluator.evaluate_predictions(baseline_preds, system_name="Baseline NMS (Current)")
    b_ov = base_eval["overall"]
    print(f"Baseline: Precision: {b_ov['precision']*100:.2f}% | Recall: {b_ov['recall']*100:.2f}% | mAP@50: {b_ov['map50']*100:.2f}% | F1: {b_ov['f1']*100:.2f}% | Detections: {sum(len(p) for p in baseline_preds.values())} | TP: {b_ov['total_tp']} | FP: {b_ov['total_fp']}")

    # Grid search across NMS variants
    test_configs = []
    for method in ["standard", "gaussian", "hybrid", "linear"]:
        for score_th in [0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40]:
            for iou_th in [0.30, 0.35, 0.40]:
                for dup_th in ([0.50, 0.55, 0.60] if method == "hybrid" else [0.55]):
                    for sigma in ([0.25, 0.35, 0.45] if method in ("gaussian", "hybrid") else [0.35]):
                        for decay in ([0.85, 0.90] if method == "linear" else [0.85]):
                            test_configs.append({
                                "method": method,
                                "score_thresh": score_th,
                                "iou_thresh": iou_th,
                                "duplicate_cutoff_iou": dup_th,
                                "sigma": sigma,
                                "score_decay": decay,
                            })

    print(f"\nSweeping {len(test_configs)} NMS & post-processing parameter configurations...")
    results = []

    for cfg in test_configs:
        preds = {}
        for img_name, d_list in raw_fused_by_img.items():
            by_class = {}
            for d in d_list:
                by_class.setdefault(d["label"], []).append({
                    "label": d["label"],
                    "class_id": d["class_id"],
                    "bbox": d["bbox"],
                    "score": d["confidence"]
                })
            kept = []
            for lbl, group in by_class.items():
                scale_map = {
                    "tiny": max(0.20, cfg["iou_thresh"] - 0.07),
                    "small": max(0.25, cfg["iou_thresh"] - 0.03),
                    "medium": cfg["iou_thresh"],
                    "large": min(0.55, cfg["iou_thresh"] + 0.08),
                }
                kept.extend(run_nms_variant(
                    group,
                    method=cfg["method"],
                    iou_thresh=cfg["iou_thresh"],
                    duplicate_cutoff_iou=cfg["duplicate_cutoff_iou"],
                    score_thresh=cfg["score_thresh"],
                    sigma=cfg["sigma"],
                    score_decay=cfg["score_decay"],
                    use_scale_aware=True,
                    scale_iou_map=scale_map
                ))
            preds[img_name] = [
                {"class_id": CLASS_NAMES.index(k["label"]), "label": k["label"], "confidence": k["score"], "bbox": k["bbox"]}
                for k in kept if k["label"] in CLASS_NAMES
            ]

        ev = evaluator.evaluate_predictions(preds, system_name="test")
        ov = ev["overall"]
        results.append({
            "cfg": cfg,
            "precision": ov["precision"],
            "recall": ov["recall"],
            "map50": ov["map50"],
            "f1": ov["f1"],
            "total_dets": sum(len(p) for p in preds.values()),
            "tp": ov["total_tp"],
            "fp": ov["total_fp"],
            "fn": ov["total_fn"],
            "gap": abs(ov["precision"] - ov["recall"]),
        })

    # Sort by F1 descending, and secondary by smaller gap and higher precision
    results.sort(key=lambda r: (r["f1"], -r["gap"], r["precision"]), reverse=True)

    print("\nTop 10 NMS Optimization Configurations:")
    print("=" * 115)
    print(f"{'Method':<9} | {'ScoreTh':<7} | {'IoUTh':<5} | {'DupIoU':<6} | {'Sigma/Decay':<11} | {'Precision':<9} | {'Recall':<8} | {'mAP@50':<8} | {'F1-Score':<8} | {'Detections':<10} | {'TP':<4} | {'FP':<4}")
    print("-" * 115)
    for r in results[:10]:
        c = r["cfg"]
        s_d = f"s={c['sigma']}" if c["method"] in ("gaussian", "hybrid") else f"d={c['score_decay']}"
        print(f"{c['method']:<9} | {c['score_thresh']:<7.2f} | {c['iou_thresh']:<5.2f} | {c['duplicate_cutoff_iou']:<6.2f} | {s_d:<11} | {r['precision']*100:<8.2f}% | {r['recall']*100:<7.2f}% | {r['map50']*100:<7.2f}% | {r['f1']*100:<7.2f}% | {r['total_dets']:<10} | {r['tp']:<4} | {r['fp']:<4}")

    out_file = PROJECT_ROOT / "#FILLERS" / "scratch" / "nms_optimization_grid_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": {
                "precision": b_ov["precision"],
                "recall": b_ov["recall"],
                "map50": b_ov["map50"],
                "f1": b_ov["f1"],
                "tp": b_ov["total_tp"],
                "fp": b_ov["total_fp"],
                "fn": b_ov["total_fn"],
                "detections": sum(len(p) for p in baseline_preds.values()),
            },
            "top_configurations": results[:20],
        }, f, indent=2)

    print(f"\nSaved optimization analysis to {out_file}")


if __name__ == "__main__":
    main()
