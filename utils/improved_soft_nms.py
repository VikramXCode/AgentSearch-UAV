#!/usr/bin/env python3
"""
Improved Soft-NMS, Class-Adaptive NMS, and Scale-Aware NMS for AgentSearch-UAV.

Provides:
1. Standard Greedy NMS
2. Linear Soft-NMS (score decays linearly with overlap)
3. Gaussian Soft-NMS (continuous Gaussian suppression)
4. Hybrid Soft-NMS (hard suppression for tile duplicate clones, soft decay for adjacent objects)
5. Class-Adaptive NMS (Standard greedy for rigid vehicles, Gaussian Soft-NMS for crowded people/pedestrians)
6. Scale-Aware IoU Thresholding (different thresholds for tiny/small/medium/large objects)
7. Cross-class and Per-class NMS modes
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Scale categories based on pixel area
SCALE_AREA_BOUNDS = {
    "tiny": (0, 32 * 32),            # < 1024 px^2
    "small": (32 * 32, 48 * 48),      # 1024 - 2304 px^2
    "medium": (48 * 48, 96 * 96),     # 2304 - 9216 px^2
    "large": (96 * 96, float("inf")), # > 9216 px^2
}

# Scale-specific default IoU thresholds tuned for aerial UAV imagery
DEFAULT_SCALE_IOU = {
    "tiny": 0.25,    # Tight NMS for tiny objects to eliminate tile duplication
    "small": 0.30,   # Balanced
    "medium": 0.35,  # Standard
    "large": 0.45,   # Looser for large objects
}

VEHICLE_CLASSES = {"car", "van", "truck", "bus", "tricycle", "awning-tricycle"}
CROWD_CLASSES = {"pedestrian", "people", "bicycle", "motor"}


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter_area

    if union <= 0.0:
        return 0.0
    return inter_area / union


def get_scale_category(box: List[float]) -> str:
    """Determine the scale category of a bounding box."""
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    area = w * h
    if area < 32 * 32:
        return "tiny"
    elif area < 48 * 48:
        return "small"
    elif area <= 96 * 96:
        return "medium"
    return "large"


class ImprovedSoftNMS:
    """High-performance Soft-NMS, Class-Adaptive NMS, and Scale-Aware NMS processor."""

    def __init__(
        self,
        method: str = "class_adaptive",  # "class_adaptive", "hybrid", "gaussian", "linear", "standard"
        iou_threshold: float = 0.32,
        duplicate_cutoff_iou: float = 0.55,
        score_threshold: float = 0.35,
        sigma: float = 0.25,
        score_decay: float = 0.85,
        use_scale_aware: bool = True,
        scale_iou_map: Optional[Dict[str, float]] = None,
    ):
        self.method = method.lower()
        self.iou_threshold = iou_threshold
        self.duplicate_cutoff_iou = duplicate_cutoff_iou
        self.score_threshold = score_threshold
        self.sigma = sigma
        self.score_decay = score_decay
        self.use_scale_aware = use_scale_aware
        self.scale_iou_map = scale_iou_map or dict(DEFAULT_SCALE_IOU)

    def get_effective_iou_threshold(self, box: List[float], base_threshold: Optional[float] = None) -> float:
        """Get scale-adapted IoU threshold for a box."""
        ref_th = base_threshold if base_threshold is not None else self.iou_threshold
        if not self.use_scale_aware:
            return ref_th
        scale = get_scale_category(box)
        offset = ref_th - self.iou_threshold
        return max(0.18, min(0.60, self.scale_iou_map.get(scale, ref_th) + offset))

    def process_detections(
        self,
        detections: List[Any],
        per_class: bool = True,
    ) -> List[Any]:
        """
        Apply Soft-NMS / NMS to a list of Detection objects.
        Supports both per-class and cross-class suppression.
        """
        if not detections:
            return []

        if per_class:
            # Group by class label
            class_groups: Dict[str, List[Any]] = {}
            for d in detections:
                label = getattr(d, "label", getattr(d, "class_name", "unknown")).lower()
                class_groups.setdefault(label, []).append(d)

            processed = []
            for label, group in class_groups.items():
                if self.method == "class_adaptive":
                    # Rigid vehicles: standard greedy NMS with tight IoU (0.30)
                    if label in VEHICLE_CLASSES:
                        processed.extend(self._suppress_group(
                            group,
                            method_override="standard",
                            iou_override=0.30
                        ))
                    else:
                        # Crowds/pedestrians: Gaussian Soft-NMS with duplicate cutoff (0.55)
                        processed.extend(self._suppress_group(
                            group,
                            method_override="hybrid",
                            iou_override=0.36
                        ))
                else:
                    processed.extend(self._suppress_group(group))

            # Sort overall output by confidence descending
            processed.sort(key=lambda d: getattr(d, "confidence", 0.0), reverse=True)
            return processed
        else:
            return self._suppress_group(detections)

    def _suppress_group(
        self,
        group: List[Any],
        method_override: Optional[str] = None,
        iou_override: Optional[float] = None,
    ) -> List[Any]:
        """Apply NMS/Soft-NMS on a single homogeneous or mixed detection group."""
        if len(group) <= 1:
            if group and getattr(group[0], "confidence", 0.0) >= self.score_threshold:
                return group
            elif not group:
                return []
            return [g for g in group if getattr(g, "confidence", 0.0) >= self.score_threshold]

        effective_method = (method_override or self.method).lower()
        base_iou = iou_override if iou_override is not None else self.iou_threshold

        # Extract items into mutable working list
        items = []
        for d in group:
            bbox = list(getattr(d, "bbox", [0, 0, 0, 0]))
            conf = float(getattr(d, "confidence", 0.0))
            items.append({
                "obj": d,
                "bbox": bbox,
                "score": conf,
            })

        # Sort by confidence descending
        items.sort(key=lambda x: x["score"], reverse=True)
        kept_detections: List[Any] = []

        while items:
            # Pop best candidate
            best = items.pop(0)
            best_bbox = best["bbox"]
            best_score = best["score"]

            if best_score < self.score_threshold:
                break

            # Update detection object confidence if it decayed
            if hasattr(best["obj"], "confidence"):
                best["obj"].confidence = best_score
            kept_detections.append(best["obj"])

            # Compute scale-adapted IoU threshold for this anchor
            effective_iou_thresh = self.get_effective_iou_threshold(best_bbox, base_threshold=base_iou)

            remaining = []
            for item in items:
                iou = compute_iou(best_bbox, item["bbox"])

                # Hard suppress extreme duplicate clones from overlapping tiles
                if iou >= self.duplicate_cutoff_iou:
                    continue

                if effective_method == "standard":
                    if iou < effective_iou_thresh:
                        remaining.append(item)
                elif effective_method == "linear":
                    if iou >= effective_iou_thresh:
                        item["score"] = item["score"] * (1.0 - iou * self.score_decay)
                    if item["score"] >= self.score_threshold:
                        remaining.append(item)
                elif effective_method == "gaussian":
                    if iou >= effective_iou_thresh:
                        weight = math.exp(-(iou * iou) / self.sigma)
                        item["score"] = item["score"] * weight
                    if item["score"] >= self.score_threshold:
                        remaining.append(item)
                elif effective_method == "hybrid":
                    if iou >= effective_iou_thresh:
                        weight = math.exp(-(iou * iou) / self.sigma)
                        item["score"] = item["score"] * weight
                    if item["score"] >= self.score_threshold:
                        remaining.append(item)
                else:
                    if iou < effective_iou_thresh:
                        remaining.append(item)

            items = remaining
            if effective_method in ("linear", "gaussian", "hybrid"):
                items.sort(key=lambda x: x["score"], reverse=True)

        return kept_detections
