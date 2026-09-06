#!/usr/bin/env python3
"""
Class Confusion & Conflict Resolution Engine for AgentSearch-UAV.

Resolves empirical aerial confusion pairs (car ↔ van, pedestrian ↔ people,
bicycle ↔ motor, truck ↔ bus, tricycle ↔ awning-tricycle) via geometric priors,
aspect-ratio heuristics, spatial clustering, and overlapping bounding-box arbitration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models.schemas import Detection


def compute_iou(box1: List[float], box2: List[float]) -> float:
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


class ClassConflictResolver:
    """Disambiguates mutually confusable VisDrone aerial categories."""

    def __init__(
        self,
        conflict_iou_threshold: float = 0.45,
        enable_geometric_rules: bool = True,
        enable_overlap_arbitration: bool = True,
    ):
        self.conflict_iou_threshold = conflict_iou_threshold
        self.enable_geometric_rules = enable_geometric_rules
        self.enable_overlap_arbitration = enable_overlap_arbitration

    def resolve_conflicts(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply full conflict resolution and overlapping box arbitration.
        """
        if len(detections) <= 1:
            return detections

        # 1. Apply geometric rules to single boxes
        if self.enable_geometric_rules:
            for d in detections:
                self._apply_single_box_heuristics(d)

        # 2. Arbitrate overlapping competing predictions
        if self.enable_overlap_arbitration:
            return self._arbitrate_overlapping_conflicts(detections)

        return detections

    def _apply_single_box_heuristics(self, detection: Detection) -> None:
        """Refine label based on aspect ratio and spatial geometry."""
        bbox = detection.bbox
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        aspect_ratio = w / h
        area = w * h
        label = detection.label.lower()

        # Heuristic 1: Bicycle vs Motor
        # Bicycles in VisDrone rarely exceed 1500 px^2; larger heavier vehicles labeled bicycle are typically motors
        if label == "bicycle" and area > 2500 and detection.confidence < 0.40:
            detection.label = "motor"
            detection.metadata["conflict_resolved"] = "bicycle_to_motor_by_area"

        # Heuristic 2: Car vs Van
        # Vans from top-down UAV views are noticeably more elongated (aspect ratio > 1.8 or < 0.55)
        # and have larger volume than standard compact cars
        if label == "car" and area > 3500 and (aspect_ratio > 1.9 or aspect_ratio < 0.52):
            # If model had marginal confidence on car, van is a strong possibility
            if 0.20 <= detection.confidence <= 0.45:
                detection.metadata["van_candidate"] = True

        # Heuristic 3: Truck vs Bus
        # Buses have high aspect ratio (long rectangular passenger compartment)
        if label == "truck" and (aspect_ratio > 2.2 or aspect_ratio < 0.45) and area > 6000:
            if detection.confidence < 0.35:
                detection.label = "bus"
                detection.metadata["conflict_resolved"] = "truck_to_bus_by_length"

    def _arbitrate_overlapping_conflicts(self, detections: List[Detection]) -> List[Detection]:
        """
        When two different classes overlap significantly (> IoU threshold),
        determine which prediction is more credible.
        """
        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        suppressed_indices = set()

        for i in range(len(sorted_dets)):
            if i in suppressed_indices:
                continue

            det_a = sorted_dets[i]
            label_a = det_a.label.lower()

            for j in range(i + 1, len(sorted_dets)):
                if j in suppressed_indices:
                    continue

                det_b = sorted_dets[j]
                label_b = det_b.label.lower()

                # If same class, let NMS handle it
                if label_a == label_b:
                    continue

                iou = compute_iou(det_a.bbox, det_b.bbox)
                if iou < self.conflict_iou_threshold:
                    continue

                # We have a high-overlap conflict between different classes!
                # Arbitration Pair 1: Car vs Van
                if {label_a, label_b} == {"car", "van"}:
                    van_det = det_a if label_a == "van" else det_b
                    car_det = det_a if label_a == "car" else det_b
                    van_idx = i if label_a == "van" else j
                    car_idx = i if label_a == "car" else j

                    w = max(1.0, van_det.bbox[2] - van_det.bbox[0])
                    h = max(1.0, van_det.bbox[3] - van_det.bbox[1])
                    ar = w / h

                    # If elongated and van confidence is competitive (> 70% of car confidence)
                    if (ar > 1.7 or ar < 0.58) and van_det.confidence >= (car_det.confidence * 0.70):
                        suppressed_indices.add(car_idx)
                    else:
                        suppressed_indices.add(van_idx)

                # Arbitration Pair 2: Pedestrian vs People
                elif {label_a, label_b} == {"pedestrian", "people"}:
                    ped_idx = i if label_a == "pedestrian" else j
                    people_idx = i if label_a == "people" else j
                    # Prefer higher confidence prediction
                    if det_a.confidence > det_b.confidence * 1.15:
                        suppressed_indices.add(j)
                    else:
                        suppressed_indices.add(people_idx)

                # Arbitration Pair 3: Bicycle vs Motor
                elif {label_a, label_b} == {"bicycle", "motor"}:
                    # Prefer the more confident detection
                    suppressed_indices.add(j)

                # Arbitration Pair 4: Tricycle vs Awning-Tricycle
                elif {label_a, label_b} == {"tricycle", "awning-tricycle"}:
                    suppressed_indices.add(j)

                else:
                    # Generic cross-class conflict: suppress lower confidence if IoU is high (> 0.60)
                    if iou > 0.60:
                        suppressed_indices.add(j)

        return [d for idx, d in enumerate(sorted_dets) if idx not in suppressed_indices]
