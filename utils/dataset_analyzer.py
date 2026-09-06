#!/usr/bin/env python3
"""
Dataset Imbalance Analyzer & Augmentation Configuration for AgentSearch-UAV.

Analyzes per-class instance distribution in VisDrone, computes Focal Loss
class weights (alpha_c), and provides minority-class augmentation configurations
for further fine-tuning pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


class DatasetImbalanceAnalyzer:
    """Computes class statistics and balanced loss weights."""

    def __init__(self, train_dir: Optional[Path] = None, val_dir: Optional[Path] = None):
        self.train_dir = train_dir or (PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-train")
        self.val_dir = val_dir or (PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val")

    def count_annotations(self, dataset_dir: Path) -> Dict[str, int]:
        """Count valid target annotations in a VisDrone dataset split."""
        ann_dir = dataset_dir / "annotations"
        counts = {name: 0 for name in CLASS_NAMES}
        if not ann_dir.exists():
            return counts

        for ann_file in ann_dir.glob("*.txt"):
            for line in ann_file.read_text(encoding="utf-8").splitlines():
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) >= 6:
                    try:
                        score = float(parts[4])
                        cat = int(float(parts[5]))
                        if 1 <= cat <= 10 and score > 0:
                            counts[CLASS_NAMES[cat - 1]] += 1
                    except (ValueError, IndexError):
                        continue
        return counts

    def compute_balanced_loss_weights(
        self,
        class_counts: Dict[str, int],
        power: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute inverse-frequency Focal Loss weights:
        alpha_c = (max_count / count_c) ** power, normalized so mean(alpha) = 1.0.
        """
        counts = np.array([max(1, class_counts.get(c, 1)) for c in CLASS_NAMES], dtype=float)
        max_c = np.max(counts)
        raw_weights = (max_c / counts) ** power
        normalized_weights = raw_weights / np.mean(raw_weights)

        return {name: round(float(w), 4) for name, w in zip(CLASS_NAMES, normalized_weights)}

    def get_augmentation_recommendations(self) -> Dict[str, Any]:
        """Recommended aerial augmentations for minority classes."""
        return {
            "minority_classes": ["awning-tricycle", "bicycle", "tricycle", "bus"],
            "recommended_augmentations": {
                "horizontal_flip_prob": 0.50,
                "vertical_flip_prob": 0.0,  # Avoid inverted UAV imagery for ground vehicles
                "scale_jitter_range": [0.8, 1.3],
                "hsv_h_jitter": 0.015,
                "hsv_s_jitter": 0.5,
                "hsv_v_jitter": 0.4,
                "moderate_rotation_degrees": 10.0,
                "minority_mosaic_oversample_ratio": 2.5,
            },
        }

    def generate_full_analysis(self) -> Dict[str, Any]:
        train_counts = self.count_annotations(self.train_dir)
        val_counts = self.count_annotations(self.val_dir)
        loss_weights = self.compute_balanced_loss_weights(train_counts)

        total_train = max(1, sum(train_counts.values()))
        total_val = max(1, sum(val_counts.values()))

        per_class_data = []
        for name in CLASS_NAMES:
            t_cnt = train_counts.get(name, 0)
            v_cnt = val_counts.get(name, 0)
            per_class_data.append({
                "class_name": name,
                "train_count": t_cnt,
                "train_share_pct": round(t_cnt / total_train * 100, 2),
                "val_count": v_cnt,
                "val_share_pct": round(v_cnt / total_val * 100, 2),
                "loss_weight_alpha": loss_weights.get(name, 1.0),
                "is_minority": t_cnt / total_train < 0.05,
            })

        return {
            "total_train_instances": total_train,
            "total_val_instances": total_val,
            "per_class": per_class_data,
            "loss_weights": loss_weights,
            "augmentation_recommendations": self.get_augmentation_recommendations(),
        }
