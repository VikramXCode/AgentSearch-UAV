#!/usr/bin/env python3
"""
Adaptive SAHI (Sliced Aided Hyper Inference) Engine for AgentSearch-UAV.

Dynamically configures patch dimensions, overlap ratios, and invocation decisions
based on input image resolution, estimated object density, and small-object distribution.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from models.schemas import Detection


class AdaptiveSAHI:
    """Adaptive sliced inference coordinator for UAV imagery."""

    def __init__(
        self,
        min_density_for_sahi: int = 5,
        min_small_ratio_for_sahi: float = 0.25,
        default_slice_width: int = 640,
        default_slice_height: int = 640,
        default_overlap: float = 0.20,
        dense_slice_width: int = 480,
        dense_slice_height: int = 480,
        dense_overlap: float = 0.28,
    ):
        self.min_density_for_sahi = min_density_for_sahi
        self.min_small_ratio_for_sahi = min_small_ratio_for_sahi
        self.default_slice_width = default_slice_width
        self.default_slice_height = default_slice_height
        self.default_overlap = default_overlap
        self.dense_slice_width = dense_slice_width
        self.dense_slice_height = dense_slice_height
        self.dense_overlap = dense_overlap

    def should_trigger_sahi(
        self,
        base_detections: List[Detection],
        image_width: int,
        image_height: int,
        force_sahi: bool = False,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Determine whether sliced inference is beneficial for this image.
        """
        if force_sahi:
            return True, "forced", {"strategy": "forced"}

        num_dets = len(base_detections)
        if num_dets == 0:
            # If base detection found nothing on a large image (> 1.5M pixels), SAHI might uncover hidden small targets
            if (image_width * image_height) >= 1_000_000:
                return True, "empty_large_image", {"strategy": "standard"}
            return False, "sparse_empty", {"strategy": "none"}

        # Calculate small object ratio
        small_count = sum(1 for d in base_detections if (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) < 32 * 32)
        small_ratio = small_count / max(1, num_dets)

        # Decision rule 1: High density scene (> 25 objects) with small objects
        if num_dets >= 25 and small_ratio >= 0.20:
            return True, "dense_small_heavy", {"strategy": "dense"}

        # Decision rule 2: High proportion of small objects (> 35%)
        if small_ratio >= self.min_small_ratio_for_sahi and num_dets >= self.min_density_for_sahi:
            return True, "small_object_heavy", {"strategy": "standard"}

        # Decision rule 3: Very large image (> 3.5M pixels)
        if (image_width * image_height) >= 3_500_000:
            return True, "ultra_high_res", {"strategy": "standard"}

        return False, "skipped_normal_scene", {"strategy": "none"}

    def generate_slices(
        self,
        image: Image.Image,
        strategy: str = "standard",
    ) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]]]:
        """
        Generate overlapping image slices and offset coordinates.
        Returns: (patches, [(x1, y1, x2, y2), ...])
        """
        w, h = image.size

        if strategy == "dense":
            slice_w, slice_h = self.dense_slice_width, self.dense_slice_height
            overlap = self.dense_overlap
        else:
            slice_w, slice_h = self.default_slice_width, self.default_slice_height
            overlap = self.default_overlap

        # Ensure slice size does not exceed image dimensions
        slice_w = min(w, slice_w)
        slice_h = min(h, slice_h)

        step_x = max(1, int(slice_w * (1.0 - overlap)))
        step_y = max(1, int(slice_h * (1.0 - overlap)))

        x_coords = list(range(0, max(1, w - slice_w + step_x), step_x))
        y_coords = list(range(0, max(1, h - slice_h + step_y), step_y))

        if not x_coords or x_coords[-1] + slice_w < w:
            x_coords.append(max(0, w - slice_w))
        if not y_coords or y_coords[-1] + slice_h < h:
            y_coords.append(max(0, h - slice_h))

        x_coords = sorted(set(min(x, max(0, w - slice_w)) for x in x_coords))
        y_coords = sorted(set(min(y, max(0, h - slice_h)) for y in y_coords))

        patches: List[Image.Image] = []
        boxes: List[Tuple[int, int, int, int]] = []

        for y in y_coords:
            for x in x_coords:
                x2 = min(x + slice_w, w)
                y2 = min(y + slice_h, h)
                x1 = max(0, x2 - slice_w)
                y1 = max(0, y2 - slice_h)

                patch = image.crop((x1, y1, x2, y2))
                patches.append(patch)
                boxes.append((x1, y1, x2, y2))

        return patches, boxes

    def run_sliced_inference(
        self,
        image: Image.Image,
        detect_fn: Callable[[List[Image.Image]], List[List[Detection]]],
        strategy: str = "standard",
    ) -> List[Detection]:
        """
        Execute adaptive sliced detection, map coordinates back to the full frame,
        and mark detections with source='sahi'.
        """
        patches, slice_boxes = self.generate_slices(image, strategy=strategy)
        if not patches:
            return []

        # Batch predict all patches
        patch_predictions = detect_fn(patches)
        global_detections: List[Detection] = []

        for (x1, y1, x2, y2), patch_dets in zip(slice_boxes, patch_predictions):
            for d in patch_dets:
                local_box = d.bbox
                # Transform to global frame coordinates
                gx1 = local_box[0] + x1
                gy1 = local_box[1] + y1
                gx2 = local_box[2] + x1
                gy2 = local_box[3] + y1

                d_copy = Detection(
                    label=d.label,
                    confidence=d.confidence,
                    bbox=[float(gx1), float(gy1), float(gx2), float(gy2)],
                    class_id=d.class_id,
                    source="sahi",
                    scale_category=d.scale_category,
                    metadata=dict(d.metadata),
                )
                global_detections.append(d_copy)

        # Apply NMS across patch detections
        if not global_detections:
            return []
            
        bboxes = [[d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]] for d in global_detections]
        scores = [d.confidence for d in global_detections]
        
        # We process NMS per class to avoid suppressing different objects
        final_detections = []
        class_ids = set(d.class_id for d in global_detections)
        
        import cv2
        for cid in class_ids:
            c_indices = [i for i, d in enumerate(global_detections) if d.class_id == cid]
            c_bboxes = [bboxes[i] for i in c_indices]
            c_scores = [scores[i] for i in c_indices]
            
            indices = cv2.dnn.NMSBoxes(c_bboxes, c_scores, score_threshold=0.05, nms_threshold=0.35)
            if len(indices) > 0:
                for idx in indices.flatten():
                    final_detections.append(global_detections[c_indices[idx]])
                    
        return final_detections
