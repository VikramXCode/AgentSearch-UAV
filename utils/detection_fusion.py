#!/usr/bin/env python3
"""
Multi-Source Detection Fusion Engine for AgentSearch-UAV.

Intelligently merges detections from Full-Image Inference, Adaptive SAHI,
and Selective Super-Resolution using Weighted Box Fusion (WBF) and confidence consensus.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


class DetectionFusionEngine:
    """Fuses multi-source detections (Full, SAHI, Enhanced ROI)."""

    def __init__(
        self,
        iou_threshold: float = 0.60,
        source_weights: Optional[Dict[str, float]] = None,
        enable_consensus_boost: bool = True,
    ):
        self.iou_threshold = iou_threshold
        self.source_weights = source_weights or {
            "full_image": 1.0,
            "sahi": 1.1,           # Slight preference for sliced detection on small targets
            "enhanced_roi": 1.15,   # Higher preference for super-resolved features
        }
        self.enable_consensus_boost = enable_consensus_boost

    def fuse(
        self,
        full_image_detections: List[Detection],
        sahi_detections: Optional[List[Detection]] = None,
        roi_detections: Optional[List[Detection]] = None,
    ) -> List[Detection]:
        """
        Fuse detections from all active sources.
        """
        all_detections: List[Detection] = []
        if full_image_detections:
            all_detections.extend(full_image_detections)
        if sahi_detections:
            all_detections.extend(sahi_detections)
        if roi_detections:
            all_detections.extend(roi_detections)

        if len(all_detections) <= 1:
            return all_detections

        # Group detections by label
        class_groups: Dict[str, List[Detection]] = {}
        for d in all_detections:
            label = d.label.lower()
            class_groups.setdefault(label, []).append(d)

        fused_results: List[Detection] = []
        for label, group in class_groups.items():
            fused_group = self._fuse_class_cluster(group, label)
            fused_results.extend(fused_group)

        # Sort overall results by confidence descending
        fused_results.sort(key=lambda d: d.confidence, reverse=True)
        return fused_results

    def _fuse_class_cluster(self, detections: List[Detection], label: str) -> List[Detection]:
        """Apply Weighted Box Fusion on a single class group."""
        if len(detections) <= 1:
            return detections

        # Sort detections by confidence descending
        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        used = [False] * len(dets)
        fused_list: List[Detection] = []

        for i in range(len(dets)):
            if used[i]:
                continue

            cluster = [dets[i]]
            used[i] = True

            for j in range(i + 1, len(dets)):
                if used[j]:
                    continue

                if compute_iou(dets[i].bbox, dets[j].bbox) >= self.iou_threshold:
                    cluster.append(dets[j])
                    used[j] = True

            if len(cluster) == 1:
                fused_list.append(cluster[0])
            else:
                # Weighted average box coordinates
                weights = [self.source_weights.get(d.source, 1.0) * d.confidence for d in cluster]
                total_w = sum(weights)

                w_x1 = sum(w * d.bbox[0] for w, d in zip(weights, cluster)) / max(1e-6, total_w)
                w_y1 = sum(w * d.bbox[1] for w, d in zip(weights, cluster)) / max(1e-6, total_w)
                w_x2 = sum(w * d.bbox[2] for w, d in zip(weights, cluster)) / max(1e-6, total_w)
                w_y2 = sum(w * d.bbox[3] for w, d in zip(weights, cluster)) / max(1e-6, total_w)

                # Confidence calculation
                max_conf = max(d.confidence for d in cluster)
                sources = sorted(list(set(d.source for d in cluster)))

                # If multiple independent sources agreed, apply slight consensus boost (max +0.05)
                if self.enable_consensus_boost and len(sources) > 1:
                    fused_conf = min(0.99, max_conf + (0.03 * (len(sources) - 1)))
                else:
                    fused_conf = max_conf

                fused_detection = Detection(
                    label=label,
                    confidence=float(fused_conf),
                    bbox=[float(w_x1), float(w_y1), float(w_x2), float(w_y2)],
                    class_id=cluster[0].class_id,
                    source="fused" if len(sources) > 1 else sources[0],
                    scale_category=cluster[0].scale_category,
                    metadata={
                        "fused_sources": sources,
                        "cluster_size": len(cluster),
                    },
                )
                fused_list.append(fused_detection)

        return fused_list
