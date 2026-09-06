#!/usr/bin/env python3
"""
Advanced Post-Processing & Class-Wise NMS Optimizer for AgentUAV.

Tests:
1. Class-adaptive NMS:
   - Rigid vehicles (car, van, truck, bus): Stricter suppression (standard NMS, IoU 0.28-0.35)
   - Crowded/non-rigid (pedestrian, people, motor, bicycle): Soft-NMS (Gaussian sigma=0.25-0.35)
2. Cross-class conflict resolution tuning (0.45 vs 0.50 vs 0.55)
3. False positive geometric filtering
"""

from __future__ import annotations

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

_current = Path(__file__).resolve().parent
PROJECT_ROOT = _current.parent.parent if _current.parent.name == "#FILLERS" else _current.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "#FILLERS"))

from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, CLASS_NAMES
from models.schemas import Detection
from utils.improved_soft_nms import compute_iou, get_scale_category
from utils.class_conflict_resolver import ClassConflictResolver
from utils.false_positive_filter import FalsePositiveFilter


def run_class_adaptive_nms(
    items: List[Dict[str, Any]],
    vehicle_method: str = "standard",
    vehicle_iou: float = 0.30,
    crowd_method: str = "gaussian",
    crowd_iou: float = 0.35,
    crowd_sigma: float = 0.25,
    global_score_thresh: float = 0.35,
    class_score_threshs: Dict[str, float] = None,
) -> List[Dict[str, Any]]:
    if not items:
        return []

    # Group by class
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        by_class.setdefault(it["label"], []).append(it)

    kept: List[Dict[str, Any]] = []

    vehicle_classes = {"car", "van", "truck", "bus", "tricycle", "awning-tricycle"}
    crowd_classes = {"pedestrian", "people", "bicycle", "motor"}

    for label, group in by_class.items():
        if len(group) <= 1:
            score_th = class_score_threshs.get(label, global_score_thresh) if class_score_threshs else global_score_thresh
            if group and group[0]["score"] >= score_th:
                kept.extend(group)
            continue

        score_th = class_score_threshs.get(label, global_score_thresh) if class_score_threshs else global_score_thresh
        method = vehicle_method if label in vehicle_classes else crowd_method
        base_iou = vehicle_iou if label in vehicle_classes else crowd_iou

        pool = sorted(group, key=lambda x: x["score"], reverse=True)
        class_kept = []

        while pool:
            best = pool.pop(0)
            if best["score"] < score_th:
                break
            class_kept.append(best)

            best_bbox = best["bbox"]
            scale = get_scale_category(best_bbox)

            # Scale-aware IoU adjustment
            if scale == "tiny":
                eff_iou = max(0.20, base_iou - 0.05)
            elif scale == "small":
                eff_iou = max(0.25, base_iou - 0.02)
            elif scale == "large":
                eff_iou = min(0.55, base_iou + 0.08)
            else:
                eff_iou = base_iou

            remaining = []
            for it in pool:
                iou = compute_iou(best_bbox, it["bbox"])

                if method == "standard":
                    if iou < eff_iou:
                        remaining.append(it)
                elif method == "gaussian":
                    if iou >= 0.55:
                        # Hard suppress extreme duplicates
                        continue
                    elif iou >= eff_iou:
                        it["score"] *= math.exp(-(iou * iou) / crowd_sigma)
                    if it["score"] >= score_th:
                        remaining.append(it)
                elif method == "linear":
                    if iou >= 0.55:
                        continue
                    elif iou >= eff_iou:
                        it["score"] *= (1.0 - iou * 0.85)
                    if it["score"] >= score_th:
                        remaining.append(it)

            pool = remaining
            if method in ("gaussian", "linear"):
                pool.sort(key=lambda x: x["score"], reverse=True)

        kept.extend(class_kept)

    return kept


def main():
    candidates = [
        PROJECT_ROOT / "#FILLERS" / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
        PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
    ]
    val_dir = next((c for c in candidates if c.exists()), candidates[0])
    evaluator = ComprehensiveEvaluator(val_dir=val_dir, iou_threshold=0.50)

    cache_file = PROJECT_ROOT / "#FILLERS" / "scratch" / "raw_fused_detections_cache.json"
    with open(cache_file, "r", encoding="utf-8") as f:
        raw_fused_by_img = json.load(f)

    print("Loaded raw fused detections from cache.")

    # Sweep combinations of class-adaptive NMS and conflict resolution
    configs = []
    for v_iou in [0.28, 0.30, 0.32, 0.35]:
        for c_iou in [0.32, 0.35, 0.38]:
            for c_sigma in [0.20, 0.25, 0.30]:
                for score_th in [0.32, 0.34, 0.35, 0.36]:
                    for conflict_iou in [0.45, 0.50, 0.55]:
                        configs.append({
                            "vehicle_iou": v_iou,
                            "crowd_iou": c_iou,
                            "crowd_sigma": c_sigma,
                            "score_thresh": score_th,
                            "conflict_iou": conflict_iou,
                        })

    print(f"Testing {len(configs)} class-adaptive NMS configurations...")
    best_res = []

    for cfg in configs:
        conflict_resolver = ClassConflictResolver(conflict_iou_threshold=cfg["conflict_iou"])
        preds = {}

        for img_name, d_list in raw_fused_by_img.items():
            # 1. Class-adaptive NMS
            items = [{
                "label": d["label"],
                "class_id": d["class_id"],
                "bbox": d["bbox"],
                "score": d["confidence"]
            } for d in d_list]

            kept = run_class_adaptive_nms(
                items,
                vehicle_method="standard",
                vehicle_iou=cfg["vehicle_iou"],
                crowd_method="gaussian",
                crowd_iou=cfg["crowd_iou"],
                crowd_sigma=cfg["crowd_sigma"],
                global_score_thresh=cfg["score_thresh"]
            )

            # Convert to Detection objects for conflict resolver
            dets = [
                Detection(
                    label=k["label"],
                    confidence=k["score"],
                    bbox=k["bbox"],
                    class_id=CLASS_NAMES.index(k["label"])
                ) for k in kept if k["label"] in CLASS_NAMES
            ]

            # 2. Cross-class conflict resolution
            resolved = conflict_resolver.resolve_conflicts(dets)

            preds[img_name] = [
                {"class_id": r.class_id, "label": r.label, "confidence": r.confidence, "bbox": r.bbox}
                for r in resolved
            ]

        ev = evaluator.evaluate_predictions(preds, system_name="adaptive_nms")
        ov = ev["overall"]
        best_res.append({
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
    best_res.sort(key=lambda r: (r["f1"], -r["gap"], r["precision"]), reverse=True)

    print("\nTop 10 Class-Adaptive NMS Configurations:")
    print("=" * 125)
    print(f"{'VehIoU':<7} | {'CrowdIoU':<8} | {'Sigma':<6} | {'ScoreTh':<7} | {'ConfIoU':<7} | {'Precision':<9} | {'Recall':<8} | {'mAP@50':<8} | {'F1-Score':<8} | {'Dets':<6} | {'TP':<4} | {'FP':<4}")
    print("-" * 125)
    for r in best_res[:10]:
        c = r["cfg"]
        print(f"{c['vehicle_iou']:<7.2f} | {c['crowd_iou']:<8.2f} | {c['crowd_sigma']:<6.2f} | {c['score_thresh']:<7.2f} | {c['conflict_iou']:<7.2f} | {r['precision']*100:<8.2f}% | {r['recall']*100:<7.2f}% | {r['map50']*100:<7.2f}% | {r['f1']*100:<7.2f}% | {r['total_dets']:<6} | {r['tp']:<4} | {r['fp']:<4}")

    best = best_res[0]
    print(f"\nBest Balanced Configuration: {best['cfg']}")
    print(f"Precision: {best['precision']*100:.2f}% | Recall: {best['recall']*100:.2f}% | F1: {best['f1']*100:.2f}% | mAP@50: {best['map50']*100:.2f}% | TP: {best['tp']} | FP: {best['fp']}")


if __name__ == "__main__":
    main()
