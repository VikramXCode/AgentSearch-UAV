"""
ColorVerifier module for high-performance two-stage attribute and color verification of object crops.
Combines ultra-fast OpenCV/HSV color space pre-filtering with batched contrastive CLIP scoring.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any, List

from models.clip_engine import CLIPEngine
from utils.search_utils import COLOR_ALIASES, normalize_label


# Calibrated HSV color boundaries: (lower, upper)
HSV_COLOR_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "red": [
        (np.array([0, 45, 35]), np.array([10, 255, 255])),
        (np.array([168, 45, 35]), np.array([180, 255, 255])),
    ],
    "blue": [
        (np.array([95, 45, 35]), np.array([140, 255, 255])),
    ],
    "green": [
        (np.array([35, 40, 35]), np.array([85, 255, 255])),
    ],
    "yellow": [
        (np.array([18, 45, 60]), np.array([35, 255, 255])),
    ],
    "orange": [
        (np.array([10, 55, 55]), np.array([22, 255, 255])),
    ],
    "white": [
        (np.array([0, 0, 155]), np.array([180, 45, 255])),
    ],
    "black": [
        (np.array([0, 0, 0]), np.array([180, 255, 65])),
    ],
    "gray": [
        (np.array([0, 0, 60]), np.array([180, 45, 175])),
    ],
    "grey": [
        (np.array([0, 0, 60]), np.array([180, 45, 175])),
    ],
    "silver": [
        (np.array([0, 0, 115]), np.array([180, 45, 225])),
    ],
    "brown": [
        (np.array([8, 60, 30]), np.array([20, 200, 140])),
    ],
    "pink": [
        (np.array([140, 35, 100]), np.array([170, 255, 255])),
    ],
    "purple": [
        (np.array([125, 45, 40]), np.array([155, 255, 255])),
    ],
    "gold": [
        (np.array([18, 60, 80]), np.array([30, 255, 220])),
    ],
}

CONTRASTIVE_COLOR_PALETTE = [
    "red",
    "white",
    "black",
    "blue",
    "silver",
    "gray",
    "yellow",
    "green",
    "brown",
    "orange",
]


class ColorVerifier:
    """Verifies color attributes on cropped object regions using fast two-stage filtering."""

    @staticmethod
    def verify_color(
        crop: Image.Image | np.ndarray,
        requested_color: str,
        target_class: str = "car",
        clip_engine: Optional[CLIPEngine] = None,
        hsv_threshold: float = 0.08,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Verify if a single crop matches the requested color using two-stage logic.
        """
        results = ColorVerifier.verify_color_batch(
            crops=[crop],
            requested_color=requested_color,
            target_class=target_class,
            clip_engine=clip_engine,
            hsv_threshold=hsv_threshold,
        )
        return results[0] if results else (False, 0.0, {})

    @staticmethod
    def verify_color_batch(
        crops: List[Image.Image | np.ndarray],
        requested_color: str,
        target_class: str = "car",
        clip_engine: Optional[CLIPEngine] = None,
        hsv_threshold: float = 0.08,
    ) -> List[Tuple[bool, float, Dict[str, Any]]]:
        """
        Fast two-stage batched color verification:
        Stage 1: Ultra-fast OpenCV/HSV color pre-filter on all crops (~0.1ms per crop).
                 Instantly rejects clear non-matches (HSV < 2%) and confirms strong matches (HSV >= 18%).
        Stage 2: Only forwards ambiguous candidates (2% <= HSV < 18%) to a single batched CLIP forward pass.
        """
        color_key = COLOR_ALIASES.get(requested_color.lower().strip(), requested_color.lower().strip())
        if not color_key:
            return [(True, 1.0, {"reason": "no_color_requested"}) for _ in crops]

        target_label = normalize_label(target_class) or "object"

        # Construct palette & prompts
        palette = list(CONTRASTIVE_COLOR_PALETTE)
        if color_key not in palette:
            palette.append(color_key)
        prompts = [f"a {c} {target_label}" for c in palette]
        target_prompt = f"a {color_key} {target_label}"

        results: List[Optional[Tuple[bool, float, Dict[str, Any]]]] = [None] * len(crops)
        pil_crops: List[Image.Image] = []
        cv_crops: List[np.ndarray] = []

        # Convert crops
        for crop in crops:
            if isinstance(crop, Image.Image):
                cpil = crop.convert("RGB")
                ccv = cv2.cvtColor(np.array(cpil), cv2.COLOR_RGB2BGR)
            elif isinstance(crop, np.ndarray):
                ccv = crop
                cpil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            else:
                ccv = np.zeros((0, 0, 3), dtype=np.uint8)
                cpil = Image.new("RGB", (1, 1))

            pil_crops.append(cpil)
            cv_crops.append(ccv)

        # ======================================================================
        # STAGE 1: Fast OpenCV/HSV Color Pre-Filter
        # ======================================================================
        import time
        t_stage1_start = time.perf_counter()
        hsv_ratios: List[float] = []
        ambiguous_indices: List[int] = []

        for idx, (cpil, ccv) in enumerate(zip(pil_crops, cv_crops)):
            if ccv.size == 0 or ccv.shape[0] < 2 or ccv.shape[1] < 2:
                results[idx] = (False, 0.0, {"error": "crop_too_small"})
                hsv_ratios.append(0.0)
                continue

            ratio = ColorVerifier.analyze_hsv_color_presence(ccv, color_key)
            hsv_ratios.append(ratio)

            # Clear Negative: virtually no color pixels (< 1.5% for red, < 2.0% general)
            if ratio < 0.015:
                results[idx] = (
                    False,
                    0.0,
                    {
                        "requested_color": color_key,
                        "hsv_pixel_ratio": float(ratio),
                        "stage": "hsv_prefilter_rejected",
                        "is_match": False,
                        "confidence": 0.0,
                    },
                )
            # Clear Positive: dominant color presence (>= 18% pixels)
            elif ratio >= 0.18:
                results[idx] = (
                    True,
                    round(min(1.0, 0.80 + float(ratio) * 0.4), 3),
                    {
                        "requested_color": color_key,
                        "hsv_pixel_ratio": float(ratio),
                        "stage": "hsv_prefilter_confirmed",
                        "is_match": True,
                        "confidence": round(min(1.0, 0.80 + float(ratio) * 0.4), 3),
                    },
                )
            else:
                # Ambiguous candidate: e.g. dark red, shadow, partial occlusion
                ambiguous_indices.append(idx)

        t_stage1_end = time.perf_counter()
        hsv_time = t_stage1_end - t_stage1_start
        ai_time = 0.0

        # ======================================================================
        # STAGE 2: Batched AI / CLIP Verification (Only for Ambiguous Crops)
        # ======================================================================
        if ambiguous_indices and clip_engine is not None:
            t_stage2_start = time.perf_counter()
            ambiguous_crops = [pil_crops[i] for i in ambiguous_indices]
            batch_scores = clip_engine.score_image_batch_against_texts(ambiguous_crops, prompts)
            ai_time = time.perf_counter() - t_stage2_start

            for amb_idx, scores in zip(ambiguous_indices, batch_scores):
                ratio = hsv_ratios[amb_idx]
                if scores:
                    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                    top_prompt, top_score = sorted_scores[0]
                    target_score = float(scores.get(target_prompt, 0.0))
                    second_score = float(sorted_scores[1][1]) if len(sorted_scores) > 1 else 0.0

                    is_top = (top_prompt == target_prompt)
                    margin = target_score - (second_score if is_top else top_score)
                else:
                    is_top = False
                    target_score = 0.0
                    margin = 0.0
                    top_prompt = ""

                # Decision Fusion
                if is_top:
                    if ratio >= 0.03 or margin >= 0.0001:
                        is_match = True
                        confidence = min(1.0, 0.75 + (margin * 10.0) + (ratio * 0.4))
                    else:
                        is_match = False
                        confidence = 0.35
                else:
                    # If CLIP picked another color, only accept if HSV is prominent
                    if ratio >= 0.12:
                        is_match = True
                        confidence = min(1.0, 0.65 + (ratio * 0.8))
                    else:
                        is_match = False
                        confidence = max(0.0, float(ratio) * 1.5)

                results[amb_idx] = (
                    bool(is_match),
                    round(float(confidence), 3),
                    {
                        "requested_color": color_key,
                        "hsv_pixel_ratio": float(ratio),
                        "clip_is_top": bool(is_top),
                        "clip_target_score": float(target_score),
                        "clip_margin": float(margin),
                        "top_color_prompt": top_prompt,
                        "stage": "clip_ai_verified",
                        "is_match": bool(is_match),
                        "confidence": round(float(confidence), 3),
                    },
                )
        elif ambiguous_indices:
            # Fallback without CLIP
            for amb_idx in ambiguous_indices:
                ratio = hsv_ratios[amb_idx]
                is_match = ratio >= 0.08
                confidence = min(1.0, 0.65 + ratio * 0.8) if is_match else max(0.0, ratio * 2.0)
                results[amb_idx] = (
                    bool(is_match),
                    round(float(confidence), 3),
                    {
                        "requested_color": color_key,
                        "hsv_pixel_ratio": float(ratio),
                        "stage": "hsv_fallback",
                        "is_match": bool(is_match),
                        "confidence": round(float(confidence), 3),
                    },
                )

        # Safety fill
        final_results = []
        for r in results:
            if r is not None:
                matched, conf, meta = r
                meta["hsv_filter_time"] = hsv_time
                meta["ai_verification_time"] = ai_time
                final_results.append((matched, conf, meta))
            else:
                final_results.append((False, 0.0, {"hsv_filter_time": hsv_time, "ai_verification_time": ai_time}))

        return final_results

    @staticmethod
    def analyze_hsv_color_presence(crop_cv: np.ndarray, color_name: str) -> float:
        """
        Analyze the presence fraction of a specific color in the central region of an object crop.
        """
        if color_name not in HSV_COLOR_RANGES:
            return 0.0

        hsv = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape

        # Sample the central 70% region to focus on the object body rather than surrounding pavement/shadows
        margin_y = int(h * 0.15)
        margin_x = int(w * 0.15)
        center = hsv[margin_y:h - margin_y, margin_x:w - margin_x] if (h > 6 and w > 6) else hsv
        total_pixels = float(center.shape[0] * center.shape[1] + 1e-6)

        mask = np.zeros((center.shape[0], center.shape[1]), dtype=np.uint8)
        for lower_b, upper_b in HSV_COLOR_RANGES[color_name]:
            mask |= cv2.inRange(center, lower_b, upper_b)

        color_pixel_count = np.count_nonzero(mask)
        return float(color_pixel_count / total_pixels)
