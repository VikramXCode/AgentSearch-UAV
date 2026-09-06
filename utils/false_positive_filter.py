#!/usr/bin/env python3
"""
False Positive Reduction Layer for AgentSearch-UAV.

Filters out noise, geometric anomalies, edge artifacts, and low-confidence clutter
without sacrificing recall on true small-scale UAV targets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models.schemas import Detection

# Minimum realistic pixel area per class for UAV aerial cameras
CLASS_MIN_AREAS = {
    "pedestrian": 12.0,      # Minimum ~3x4 px
    "people": 12.0,
    "bicycle": 16.0,
    "car": 36.0,             # A car should be at least ~6x6 px
    "van": 48.0,
    "truck": 64.0,
    "tricycle": 25.0,
    "awning-tricycle": 25.0,
    "bus": 80.0,
    "motor": 16.0,
}


class FalsePositiveFilter:
    """Filters spurious, distorted, and noisy detections."""

    def __init__(
        self,
        min_width: float = 3.0,
        min_height: float = 3.0,
        min_area: float = 9.0,
        min_aspect_ratio: float = 0.06,
        max_aspect_ratio: float = 16.0,
        edge_margin_ratio: float = 0.002,
        class_min_areas: Optional[Dict[str, float]] = None,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.edge_margin_ratio = edge_margin_ratio
        self.class_min_areas = class_min_areas or dict(CLASS_MIN_AREAS)

    def filter_detections(
        self,
        detections: List[Detection],
        image_width: int,
        image_height: int,
    ) -> List[Detection]:
        """Apply geometric and contextual validation to a list of detections."""
        if not detections:
            return []

        edge_x_min = self.edge_margin_ratio * image_width
        edge_x_max = (1.0 - self.edge_margin_ratio) * image_width
        edge_y_min = self.edge_margin_ratio * image_height
        edge_y_max = (1.0 - self.edge_margin_ratio) * image_height

        valid_detections = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h

            # 1. Absolute geometric bounds check
            if w < self.min_width or h < self.min_height or area < self.min_area:
                continue

            # 2. Aspect ratio check
            aspect_ratio = w / max(1.0, h)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            # 3. Class-specific minimum area check
            label = d.label.lower()
            min_class_area = self.class_min_areas.get(label, self.min_area)
            if area < min_class_area:
                continue

            # 4. Truncated edge artifact check
            # Discard low confidence detections (< 0.20) that hit the extreme frame borders
            is_touching_edge = (x1 <= edge_x_min or x2 >= edge_x_max or y1 <= edge_y_min or y2 >= edge_y_max)
            if is_touching_edge and d.confidence < 0.18:
                continue

            valid_detections.append(d)

        return valid_detections
