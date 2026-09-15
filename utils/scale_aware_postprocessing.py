#!/usr/bin/env python3
"""
Scale-Aware Detection Post-Processing for AgentSearch-UAV.

Categorizes detections by geometric footprint (Tiny, Small, Medium, Large)
and applies scale-calibrated filtering rules to boost small-object recall
while suppressing spurious false alarms on larger targets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models.schemas import Detection

# Scale area boundaries (in pixels squared)
DEFAULT_SCALE_BOUNDS = {
    "tiny": 24 * 24,       # < 576 px^2
    "small": 32 * 32,      # 576 - 1024 px^2
    "medium": 96 * 96,     # 1024 - 9216 px^2
    # large: > 9216 px^2
}


class ScaleAwarePostProcessor:
    """Scale-aware categorization and adaptive filtering."""

    def __init__(
        self,
        tiny_max_area: int = 576,
        small_max_area: int = 1024,
        medium_max_area: int = 9216,
        tiny_conf_discount: float = 0.05,
        large_conf_boost: float = 0.05,
    ):
        self.tiny_max_area = tiny_max_area
        self.small_max_area = small_max_area
        self.medium_max_area = medium_max_area
        self.tiny_conf_discount = tiny_conf_discount
        self.large_conf_boost = large_conf_boost

    def classify_scale(self, box: List[float]) -> str:
        """Classify bounding box into tiny, small, medium, or large."""
        w = max(0.0, box[2] - box[0])
        h = max(0.0, box[3] - box[1])
        area = w * h

        if area < self.tiny_max_area:
            return "tiny"
        elif area < self.small_max_area:
            return "small"
        elif area <= self.medium_max_area:
            return "medium"
        else:
            return "large"

    def process(
        self,
        detections: List[Detection],
        class_thresholds: Optional[Dict[str, float]] = None,
    ) -> List[Detection]:
        """
        Enrich detections with scale categories and apply scale-calibrated tolerance.
        """
        if not detections:
            return []

        processed = []
        for d in detections:
            scale = self.classify_scale(d.bbox)
            d.scale_category = scale

            # If class thresholds provided, apply scale-adjusted threshold
            if class_thresholds:
                label = d.label.lower()
                base_thresh = class_thresholds.get(label, 0.25)

                if scale == "tiny":
                    effective_thresh = max(0.08, base_thresh - self.tiny_conf_discount)
                elif scale == "small":
                    effective_thresh = max(0.10, base_thresh - (self.tiny_conf_discount * 0.5))
                elif scale == "large":
                    effective_thresh = min(0.60, base_thresh + self.large_conf_boost)
                else:
                    effective_thresh = base_thresh

                if d.confidence >= effective_thresh:
                    processed.append(d)
            else:
                processed.append(d)

        return processed
