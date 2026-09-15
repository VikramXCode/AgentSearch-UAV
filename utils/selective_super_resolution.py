#!/usr/bin/env python3
"""
Selective ROI Super-Resolution for AgentSearch-UAV.

Identifies challenging, low-confidence, or tiny-object candidate regions,
applies localized super-resolution to bounded ROIs only, re-detects objects,
and maps coordinates back to the full frame with source='enhanced_roi'.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from models.schemas import Detection
from models.super_resolution import SuperResolutionEngine


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


class SelectiveSuperResolution:
    """Selective patch-level super-resolution enhancement."""

    def __init__(
        self,
        scale: int = 2,
        max_rois_per_frame: int = 4,
        low_conf_threshold: float = 0.35,
        tiny_area_threshold: float = 576.0,  # 24x24 px
        roi_padding_ratio: float = 0.25,
        min_roi_size: int = 128,
        max_roi_size: int = 640,
    ):
        self.scale = scale
        self.max_rois_per_frame = max_rois_per_frame
        self.low_conf_threshold = low_conf_threshold
        self.tiny_area_threshold = tiny_area_threshold
        self.roi_padding_ratio = roi_padding_ratio
        self.min_roi_size = min_roi_size
        self.max_roi_size = max_roi_size
        self.sr_engine = SuperResolutionEngine()

    def identify_rois(
        self,
        detections: List[Detection],
        image_width: int,
        image_height: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Identify candidate ROIs needing super-resolution enhancement.
        """
        candidates = []
        for d in detections:
            w = max(0.0, d.bbox[2] - d.bbox[0])
            h = max(0.0, d.bbox[3] - d.bbox[1])
            area = w * h

            # Target low-confidence or tiny detections
            if d.confidence <= self.low_conf_threshold or area <= self.tiny_area_threshold:
                candidates.append(d.bbox)

        if not candidates:
            return []

        # Expand candidate boxes by padding
        raw_rois = []
        for box in candidates:
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            pad_w = bw * self.roi_padding_ratio
            pad_h = bh * self.roi_padding_ratio

            rx1 = max(0, int(box[0] - pad_w))
            ry1 = max(0, int(box[1] - pad_h))
            rx2 = min(image_width, int(box[2] + pad_w))
            ry2 = min(image_height, int(box[3] + pad_h))

            # Ensure minimum ROI dimension
            rw = rx2 - rx1
            rh = ry2 - ry1
            if rw < self.min_roi_size:
                diff = (self.min_roi_size - rw) // 2
                rx1 = max(0, rx1 - diff)
                rx2 = min(image_width, rx2 + diff)
            if rh < self.min_roi_size:
                diff = (self.min_roi_size - rh) // 2
                ry1 = max(0, ry1 - diff)
                ry2 = min(image_height, ry2 + diff)

            raw_rois.append([rx1, ry1, rx2, ry2])

        # Cluster/merge heavily overlapping ROIs
        merged_rois = self._merge_rois(raw_rois)

        # Cap to max_rois_per_frame
        return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in merged_rois[: self.max_rois_per_frame]]

    def _merge_rois(self, rois: List[List[int]]) -> List[List[int]]:
        """Merge nearby/overlapping ROIs."""
        if len(rois) <= 1:
            return rois

        merged = []
        used = [False] * len(rois)

        for i in range(len(rois)):
            if used[i]:
                continue
            cur = list(rois[i])
            used[i] = True

            for j in range(i + 1, len(rois)):
                if used[j]:
                    continue
                # If ROIs overlap or touch closely
                if compute_iou(cur, rois[j]) > 0.15:
                    cur[0] = min(cur[0], rois[j][0])
                    cur[1] = min(cur[1], rois[j][1])
                    cur[2] = max(cur[2], rois[j][2])
                    cur[3] = max(cur[3], rois[j][3])
                    used[j] = True

            merged.append(cur)

        return merged

    def process_image(
        self,
        image: Image.Image,
        base_detections: List[Detection],
        detect_fn: Callable[[List[Image.Image]], List[List[Detection]]],
    ) -> List[Detection]:
        """
        Crop candidate ROIs, apply super-resolution, detect, and project back to original coordinates.
        """
        w, h = image.size
        rois = self.identify_rois(base_detections, w, h)
        if not rois:
            return []

        enhanced_patches: List[Image.Image] = []
        roi_meta: List[Tuple[int, int, int, int]] = []

        for (rx1, ry1, rx2, ry2) in rois:
            roi_crop = image.crop((rx1, ry1, rx2, ry2))
            rw, rh = roi_crop.size

            # Upscale ROI 2x using Lanczos
            upscaled = roi_crop.resize((rw * self.scale, rh * self.scale), Image.Resampling.LANCZOS)
            enhanced_patches.append(upscaled)
            roi_meta.append((rx1, ry1, rx2, ry2))

        # Re-detect on enhanced ROIs
        patch_predictions = detect_fn(enhanced_patches)
        enhanced_detections: List[Detection] = []

        for (rx1, ry1, rx2, ry2), patch_dets in zip(roi_meta, patch_predictions):
            scale_factor = 1.0 / self.scale
            for d in patch_dets:
                lx1, ly1, lx2, ly2 = d.bbox
                # Scale back from 2x space and add ROI offset
                gx1 = rx1 + (lx1 * scale_factor)
                gy1 = ry1 + (ly1 * scale_factor)
                gx2 = rx1 + (lx2 * scale_factor)
                gy2 = ry1 + (ly2 * scale_factor)

                d_copy = Detection(
                    label=d.label,
                    confidence=d.confidence,
                    bbox=[float(gx1), float(gy1), float(gx2), float(gy2)],
                    class_id=d.class_id,
                    source="enhanced_roi",
                    scale_category=d.scale_category,
                    metadata=dict(d.metadata),
                )
                enhanced_detections.append(d_copy)

        return enhanced_detections
