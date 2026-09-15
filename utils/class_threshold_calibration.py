#!/usr/bin/env python3
"""
Class-Specific Confidence Threshold Calibration for AgentSearch-UAV.

Replaces uniform global confidence thresholds with empirically calibrated,
class-specific operating thresholds derived from validation precision-recall-F1 curves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default VisDrone 10 classes
CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

# Baseline fallback thresholds (empirically derived from Phase 1 F1 optimization)
DEFAULT_CLASS_THRESHOLDS: Dict[str, float] = {
    "pedestrian": 0.28,
    "people": 0.20,
    "bicycle": 0.22,
    "car": 0.34,
    "van": 0.26,
    "truck": 0.28,
    "tricycle": 0.15,
    "awning-tricycle": 0.14,
    "bus": 0.26,
    "motor": 0.22,
}


def compute_box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter_area

    iou = np.zeros_like(inter_area)
    valid = union > 0
    iou[valid] = inter_area[valid] / union[valid]
    return iou


class ClassThresholdCalibrator:
    """Calibrates and manages per-class confidence thresholds."""

    def __init__(self, thresholds_path: Optional[Path] = None):
        self.thresholds_path = thresholds_path or (PROJECT_ROOT / "configs" / "class_thresholds.json")
        self.class_thresholds = dict(DEFAULT_CLASS_THRESHOLDS)
        self.load()

    def load(self) -> Dict[str, float]:
        """Load calibrated thresholds from disk if available."""
        if self.thresholds_path.exists():
            try:
                data = json.loads(self.thresholds_path.read_text(encoding="utf-8"))
                if "thresholds" in data:
                    self.class_thresholds.update(data["thresholds"])
                elif isinstance(data, dict):
                    self.class_thresholds.update({k: float(v) for k, v in data.items() if isinstance(v, (int, float))})
            except Exception as e:
                print(f"[ClassThresholdCalibrator] Warning: Failed to load {self.thresholds_path}: {e}")
        return self.class_thresholds

    def save(self, calibration_report: Optional[Dict[str, Any]] = None) -> Path:
        """Save calibrated thresholds to JSON config."""
        self.thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1.0",
            "thresholds": self.class_thresholds,
            "calibration_details": calibration_report or {},
        }
        self.thresholds_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[ClassThresholdCalibrator] Saved calibrated thresholds to: {self.thresholds_path}")
        return self.thresholds_path

    def get_threshold(self, class_name: str, fallback: float = 0.25) -> float:
        """Get calibrated threshold for a specific class with safe fallback."""
        norm_name = class_name.strip().lower()
        if norm_name in self.class_thresholds:
            return self.class_thresholds[norm_name]
        # Canonical mappings
        alias_map = {
            "person": "pedestrian",
            "pedestrians": "pedestrian",
            "cars": "car",
            "automobile": "car",
            "motorcycle": "motor",
            "motorbike": "motor",
            "bike": "bicycle",
            "trucks": "truck",
            "buses": "bus",
            "vans": "van",
        }
        mapped = alias_map.get(norm_name, norm_name)
        return self.class_thresholds.get(mapped, fallback)

    def filter_detections_by_class_threshold(
        self,
        detections: List[Any],
        global_override: Optional[float] = None,
    ) -> List[Any]:
        """Filter a list of Detection objects according to per-class thresholds."""
        if global_override is not None:
            return [d for d in detections if getattr(d, "confidence", 0.0) >= global_override]

        filtered = []
        for d in detections:
            label = getattr(d, "label", getattr(d, "class_name", ""))
            score = getattr(d, "confidence", 0.0)
            threshold = self.get_threshold(label)
            if score >= threshold:
                filtered.append(d)
        return filtered

    @staticmethod
    def calibrate_from_predictions(
        predictions: Dict[str, List[Dict[str, Any]]],
        ground_truths: Dict[str, List[Dict[str, Any]]],
        iou_threshold: float = 0.50,
        min_precision: float = 0.20,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Calibrate optimal F1 thresholds per class from actual predictions and ground truths.
        """
        eval_images = [img for img in predictions.keys() if img in ground_truths]
        num_classes = len(CLASS_NAMES)
        name_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}

        gt_by_img_cls: Dict[Tuple[str, int], List[np.ndarray]] = {}
        total_gt = np.zeros(num_classes, dtype=int)

        for img in eval_images:
            for g in ground_truths[img]:
                c = g["class_id"]
                if 0 <= c < num_classes:
                    total_gt[c] += 1
                    key = (img, c)
                    if key not in gt_by_img_cls:
                        gt_by_img_cls[key] = []
                    gt_by_img_cls[key].append(np.array(g["bbox"], dtype=np.float32))

        preds_by_cls: Dict[int, List[Dict[str, Any]]] = {c: [] for c in range(num_classes)}
        for img in eval_images:
            for p in predictions.get(img, []):
                c = p["class_id"]
                if 0 <= c < num_classes:
                    preds_by_cls[c].append({
                        "img": img,
                        "score": float(p["confidence"]),
                        "bbox": np.array(p["bbox"], dtype=np.float32),
                    })

        calibrated_thresholds: Dict[str, float] = {}
        calibration_report: Dict[str, Any] = {}
        conf_grid = np.linspace(0.04, 0.95, 92)

        for c in range(num_classes):
            c_name = CLASS_NAMES[c]
            c_preds = preds_by_cls[c]
            n_gt = int(total_gt[c])

            if n_gt == 0 or len(c_preds) == 0:
                calibrated_thresholds[c_name] = DEFAULT_CLASS_THRESHOLDS.get(c_name, 0.25)
                calibration_report[c_name] = {"best_f1": 0.0, "threshold": calibrated_thresholds[c_name]}
                continue

            best_f1 = -1.0
            best_conf = 0.25
            best_prec = 0.0
            best_rec = 0.0
            best_tp = 0
            best_fp = 0
            best_fn = 0

            for conf in conf_grid:
                sub_preds = [p for p in c_preds if p["score"] >= conf]
                if not sub_preds:
                    continue

                img_to_indices: Dict[str, List[int]] = {}
                for idx, p in enumerate(sub_preds):
                    img_to_indices.setdefault(p["img"], []).append(idx)

                sub_boxes = np.array([p["bbox"] for p in sub_preds], dtype=np.float32)
                matched_preds = np.zeros(len(sub_preds), dtype=bool)

                for (img, cls_id), gt_boxes_list in gt_by_img_cls.items():
                    if cls_id != c or img not in img_to_indices:
                        continue

                    gt_boxes = np.array(gt_boxes_list, dtype=np.float32)
                    p_indices = img_to_indices[img]
                    p_boxes = sub_boxes[p_indices]

                    iou_mat = compute_box_iou_matrix(p_boxes, gt_boxes)
                    matched_gt = set()

                    for local_i, global_i in enumerate(p_indices):
                        ious = iou_mat[local_i]
                        best_g = -1
                        best_iou = 0.0
                        for g_i, iou_val in enumerate(ious):
                            if iou_val > best_iou and g_i not in matched_gt:
                                best_iou = iou_val
                                best_g = g_i

                        if best_iou >= iou_threshold and best_g >= 0:
                            matched_preds[global_i] = True
                            matched_gt.add(best_g)

                tp = int(np.sum(matched_preds))
                fp = int(len(sub_preds) - tp)
                fn = int(n_gt - tp)

                p_val = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                r_val = float(tp / n_gt) if n_gt > 0 else 0.0
                f1_val = float(2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0

                if f1_val > best_f1 and p_val >= min_precision:
                    best_f1 = f1_val
                    best_conf = round(float(conf), 2)
                    best_prec = round(p_val, 4)
                    best_rec = round(r_val, 4)
                    best_tp = tp
                    best_fp = fp
                    best_fn = fn

            calibrated_thresholds[c_name] = best_conf
            calibration_report[c_name] = {
                "calibrated_threshold": best_conf,
                "best_f1": round(best_f1, 4),
                "precision": best_prec,
                "recall": best_rec,
                "tp": best_tp,
                "fp": best_fp,
                "fn": best_fn,
                "gt_count": n_gt,
            }

        return calibrated_thresholds, calibration_report


def load_class_thresholds(config_path: Optional[Path] = None) -> Dict[str, float]:
    """Helper to get calibrated class thresholds dictionary."""
    calibrator = ClassThresholdCalibrator(config_path)
    return calibrator.class_thresholds
