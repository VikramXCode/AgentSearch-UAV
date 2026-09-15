from __future__ import annotations

import os
import time
from statistics import mean

from PIL import Image

from models.clip_engine import CLIPEngine
from models.color_verifier import ColorVerifier
from models.schemas import Detection
from utils.search_utils import (
    build_verification_prompts,
    canonicalize_target,
    normalize_text,
)
from workflows.state import AgentState


class VerificationAgent:
    """
    Verification Agent:
    Validates detector candidates against target semantics, spatial size criteria,
    color profiles (via ColorVerifier HSV + CLIP), and semantic CLIP alignment.
    """

    # Conservative starting calibration values
    MIN_CLIP_SIMILARITY = 0.20
    MIN_ATTRIBUTE_MARGIN = 0.01

    def __init__(self):
        self.clip_engine = None

    def run(self, state: AgentState) -> AgentState:
        print("\n==============================")
        print("  VERIFICATION AGENT")
        print("==============================")

        t_start = time.perf_counter()

        if self.clip_engine is None:
            self.clip_engine = CLIPEngine.shared()

        image_path = (
            state.detection.processed_image_path
            or state.detection.image_path
        )

        verified, color_filter_time, ai_verification_time = self._filter_detections(
            state,
            image_path,
        )

        state.verification.verified_objects = [
            {
                "label": detection.label,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
            }
            for detection in verified
        ]

        state.verification.confidence_score = (
            mean([detection.confidence for detection in verified]) if verified else 0.0
        )
        state.verification.color_filter_time = color_filter_time
        state.verification.ai_verification_time = ai_verification_time
        state.verification.total_verification_time = time.perf_counter() - t_start

        print()
        print(f"Verified objects: {len(verified)}")
        print(f"Verification confidence: {state.verification.confidence_score:.3f}")
        print(f"Color filtering time: {state.verification.color_filter_time:.3f}s")
        print(f"AI verification time: {state.verification.ai_verification_time:.3f}s")
        print(f"Total verification time: {state.verification.total_verification_time:.3f}s")

        return state

    def _filter_detections(
        self,
        state: AgentState,
        image_path: str | None,
    ) -> tuple[list[Detection], float, float]:
        if not image_path:
            image_path = state.detection.image_path

        if not image_path or not os.path.exists(image_path):
            if image_path:
                print(f"Verification image does not exist: {image_path}")
            return [], 0.0, 0.0

        color_filter_time = 0.0
        ai_verification_time = 0.0

        requested_target = canonicalize_target(state.query.target)
        attributes = state.query.attributes
        requested_size = attributes.get("size", "").strip().lower()
        requested_color = attributes.get("color", "").strip().lower()
        quantity = state.query.quantity or attributes.get("quantity", "all")

        image = Image.open(image_path).convert("RGB")
        prompts = build_verification_prompts(state.query.target, attributes)
        if not prompts:
            prompts = [requested_target]
        requested_prompt = prompts[0]

        # ----------------------------------
        # Stage 1 — Preliminary candidate filtering (confidence, target, size, crop)
        # ----------------------------------
        valid_candidates = []
        crops = []

        for item in state.detection.objects_found:
            detection = Detection(
                label=item["label"],
                confidence=float(item["confidence"]),
                bbox=[float(value) for value in item["bbox"]],
            )

            if detection.confidence < state.strategy.confidence_threshold:
                continue

            label_target = canonicalize_target(detection.label)
            target_similarity = self._target_similarity(requested_target, label_target)
            if target_similarity <= state.strategy.target_similarity_threshold:
                continue

            size_score = self._size_score(detection.bbox, image.size, requested_size) if requested_size else 1.0
            if requested_size and not self._matches_size(detection.bbox, image.size, requested_size) and size_score < 0.30:
                print(f"Rejected candidate {detection.label} (size mismatch for '{requested_size}'): size_score={size_score:.3f}")
                continue

            crop = self._crop_image_with_context(image, detection.bbox, padding_ratio=0.10)
            if crop is None:
                continue

            valid_candidates.append({
                "detection": detection,
                "label_target": label_target,
                "target_similarity": target_similarity,
                "size_score": size_score,
                "crop": crop,
            })
            crops.append(crop)

        if not valid_candidates:
            return [], 0.0, 0.0

        # ----------------------------------
        # Stage 2 — Attribute / Color Verification
        # ----------------------------------
        surviving_candidates = []

        if requested_color:
            t_col_start = time.perf_counter()
            color_results = ColorVerifier.verify_color_batch(
                crops=crops,
                requested_color=requested_color,
                target_class=requested_target,
                clip_engine=self.clip_engine,
            )
            color_filter_time = time.perf_counter() - t_col_start
            if color_results and "ai_verification_time" in color_results[0][2]:
                ai_verification_time += color_results[0][2]["ai_verification_time"]

            for item, (color_matched, color_confidence, color_details) in zip(valid_candidates, color_results):
                det = item["detection"]
                if not color_matched:
                    print(
                        f"Rejected candidate {det.label} (color mismatch for '{requested_color}'): "
                        f"hsv_ratio={color_details.get('hsv_pixel_ratio', 0):.3f}, "
                        f"stage={color_details.get('stage')}"
                    )
                    continue

                print(
                    f"Verified color '{requested_color}' for candidate {det.label}: "
                    f"conf={color_confidence:.3f}, stage={color_details.get('stage')}, "
                    f"hsv_ratio={color_details.get('hsv_pixel_ratio', 0):.3f}"
                )
                item["color_confidence"] = color_confidence
                item["color_details"] = color_details
                surviving_candidates.append(item)
        else:
            for item in valid_candidates:
                item["color_confidence"] = 1.0
                item["color_details"] = {}
                surviving_candidates.append(item)

        if not surviving_candidates:
            return [], color_filter_time, ai_verification_time

        # ----------------------------------
        # Stage 3 — General semantic CLIP verification
        # ----------------------------------
        has_attributes = bool(attributes.get("color") or attributes.get("size"))

        if not requested_color and prompts:
            t_ai_start = time.perf_counter()
            surviving_crops = [item["crop"] for item in surviving_candidates]
            score_results = self.clip_engine.score_images_against_texts(surviving_crops, prompts)
            ai_verification_time += time.perf_counter() - t_ai_start

            filtered_after_clip = []
            for item, scores in zip(surviving_candidates, score_results):
                if not scores:
                    continue

                requested_score = scores.get(requested_prompt, -1.0)
                alternatives = [score for prompt, score in scores.items() if prompt != requested_prompt]
                best_alternative = max(alternatives) if alternatives else -1.0
                margin = requested_score - best_alternative
                selected_prompt, selected_score = max(scores.items(), key=lambda x: x[1])

                print()
                print(f"Candidate: {item['detection'].label} {item['detection'].confidence:.3f}")
                print(f"Requested: {requested_prompt} {requested_score:.3f}")
                print(f"Best match: {selected_prompt} {selected_score:.3f}")
                print(f"Attribute margin: {margin:.3f}")

                # CLIP acceptance checks
                if requested_score < self.MIN_CLIP_SIMILARITY:
                    print("Rejected: CLIP similarity too low")
                    continue

                if has_attributes:
                    if selected_prompt != requested_prompt:
                        print("Rejected: another attribute prompt matched better")
                        continue
                    if margin < self.MIN_ATTRIBUTE_MARGIN:
                        print("Rejected: attribute evidence too weak")
                        continue

                item["best_prompt"] = selected_prompt
                item["clip_score"] = float(requested_score)
                item["clip_margin"] = float(margin)
                filtered_after_clip.append(item)
            surviving_candidates = filtered_after_clip
        else:
            for item in surviving_candidates:
                item["best_prompt"] = f"a {requested_color} {requested_target}" if requested_color else requested_target
                item["clip_score"] = item["color_confidence"]
                item["clip_margin"] = 0.0

        if not surviving_candidates:
            return [], color_filter_time, ai_verification_time

        # ----------------------------------
        # Stage 4 — Scoring and Ranking
        # ----------------------------------
        final_candidates = []
        for item in surviving_candidates:
            det = item["detection"]
            label_target = item["label_target"]
            target_similarity = item["target_similarity"]
            color_confidence = item["color_confidence"]
            clip_score = item["clip_score"]
            best_prompt = item["best_prompt"]
            size_score = item.get("size_score", 1.0)

            attribute_score = (
                color_confidence if requested_color
                else self._attribute_similarity(best_prompt, requested_target, requested_color, requested_size)
            )

            detector_weight = 0.40
            color_weight = 0.30 if requested_color else 0.0
            clip_weight = 0.15 if requested_color else 0.35
            target_weight = 0.15 if requested_color else 0.20
            size_weight = 0.05

            final_score = (
                det.confidence * detector_weight
                + color_confidence * color_weight
                + clip_score * clip_weight
                + target_similarity * target_weight
                + size_score * size_weight
            )

            clip_quality = max(0.0, min(1.0, (clip_score + 1.0) / 2.0))
            if requested_color:
                verified_confidence = round(min(1.0, det.confidence * 0.5 + color_confidence * 0.5), 3)
            else:
                verified_confidence = round(min(1.0, 0.70 * det.confidence + 0.30 * clip_quality), 3)

            if verified_confidence < state.strategy.confidence_threshold:
                print(
                    f"Rejected candidate {det.label} (verified confidence "
                    f"{verified_confidence:.3f} < {state.strategy.confidence_threshold:.2f})"
                )
                continue

            verified_label = f"{requested_color} {label_target}" if requested_color else det.label
            print(
                f"Accepted candidate {verified_label}: "
                f"final_score={final_score:.3f}, det_conf={det.confidence:.3f}, "
                f"ver_conf={verified_confidence:.3f}"
            )

            final_candidates.append(
                {
                    "detection": Detection(
                        label=verified_label,
                        confidence=verified_confidence,
                        bbox=det.bbox,
                    ),
                    "final_score": final_score,
                    "clip_similarity": clip_score,
                    "clip_margin": item.get("clip_margin", 0.0),
                    "target_similarity": target_similarity,
                    "attribute_score": attribute_score,
                    "size_score": size_score,
                    "best_prompt": best_prompt,
                }
            )

        # ----------------------------------
        # Stage 5 — Quantity handling
        # ----------------------------------
        if quantity == "one" and len(final_candidates) > 1:
            final_candidates = [max(final_candidates, key=lambda item: (item["final_score"], item["detection"].confidence))]
        elif state.query.quantity_value is not None and len(final_candidates) > state.query.quantity_value:
            final_candidates = sorted(final_candidates, key=lambda item: (item["final_score"], item["detection"].confidence), reverse=True)[:state.query.quantity_value]
        else:
            final_candidates.sort(key=lambda item: (item["final_score"], item["detection"].confidence), reverse=True)

        return [candidate["detection"] for candidate in final_candidates], color_filter_time, ai_verification_time

    @staticmethod
    def _target_similarity(requested_target: str, candidate_target: str) -> float:
        if not requested_target or not candidate_target:
            return 0.0
        if requested_target == candidate_target:
            return 1.0
        if requested_target == "car" and candidate_target in {"car", "vehicle", "sedan", "van", "suv", "automobile", "jeep", "taxi"}:
            return 0.92
        if requested_target == "person" and candidate_target in {"person", "pedestrian", "people", "human", "man", "woman"}:
            return 0.95
        if requested_target == "motorcycle" and candidate_target in {"motorcycle", "motorbike", "bike", "motor"}:
            return 0.95
        if requested_target == "bicycle" and candidate_target in {"bicycle", "bike"}:
            return 0.95
        if requested_target == "truck" and candidate_target in {"truck", "lorry", "vehicle"}:
            return 0.88
        if requested_target == "tricycle" and candidate_target in {"tricycle", "awning-tricycle", "awning tricycle"}:
            return 0.95
        return 0.0

    @staticmethod
    def _attribute_similarity(best_prompt: str, requested_target: str, requested_color: str, requested_size: str) -> float:
        score = 0.0
        best_prompt = normalize_text(best_prompt)
        target_in_prompt = requested_target in best_prompt
        if target_in_prompt:
            score += 0.55

        if requested_color:
            if requested_color in best_prompt:
                score += 0.35
            else:
                score += 0.05

        if requested_size:
            if requested_size in best_prompt:
                score += 0.10

        return min(score, 1.0)

    @staticmethod
    def _size_score(bbox: list[float], image_size: tuple[int, int], requested_size: str) -> float:
        if not requested_size:
            return 1.0

        image_width, image_height = image_size
        box_width = max(1.0, bbox[2] - bbox[0])
        box_height = max(1.0, bbox[3] - bbox[1])
        relative_area = (box_width * box_height) / float(max(1, image_width * image_height))

        if requested_size in {"small", "tiny", "little"}:
            return 1.0 if relative_area <= 0.12 else max(0.45, 1.0 - abs(relative_area - 0.08) / 0.20)
        if requested_size == "medium":
            return 1.0 if 0.08 < relative_area < 0.2 else max(0.50, 1.0 - abs(relative_area - 0.14) / 0.20)
        if requested_size in {"large", "big"}:
            return 1.0 if relative_area >= 0.2 else max(0.45, 1.0 - abs(relative_area - 0.25) / 0.30)
        return 1.0

    @staticmethod
    def _matches_size(
        bbox: list[float],
        image_size: tuple[int, int],
        requested_size: str,
    ) -> bool:
        image_width, image_height = image_size
        box_width = max(1.0, bbox[2] - bbox[0])
        box_height = max(1.0, bbox[3] - bbox[1])
        relative_area = (box_width * box_height) / float(max(1, image_width * image_height))

        if requested_size in {"small", "tiny", "little"}:
            return relative_area <= 0.08
        if requested_size == "medium":
            return 0.08 < relative_area < 0.20
        if requested_size in {"large", "big"}:
            return relative_area >= 0.20
        return True

    @staticmethod
    def _crop_image_with_context(
        image: Image.Image,
        bbox: list[float],
        padding_ratio: float = 0.10,
    ) -> Image.Image | None:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return None

        pad_x = width * padding_ratio
        pad_y = height * padding_ratio

        x1 = int(max(0, x1 - pad_x))
        y1 = int(max(0, y1 - pad_y))
        x2 = int(min(image.width, x2 + pad_x))
        y2 = int(min(image.height, y2 + pad_y))

        if x2 <= x1 or y2 <= y1:
            return None

        return image.crop((x1, y1, x2, y2))