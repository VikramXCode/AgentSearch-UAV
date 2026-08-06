import os
from statistics import mean

from PIL import Image

from models.clip_engine import CLIPEngine
from models.schemas import Detection
from utils.search_utils import (
    build_verification_prompts,
    canonicalize_target,
)
from workflows.state import AgentState


class VerificationAgent:

    # These are deliberately conservative starting values.
    # They should eventually be calibrated using validation data.
    MIN_CLIP_SIMILARITY = 0.20
    MIN_ATTRIBUTE_MARGIN = 0.01

    def __init__(self):
        self.clip_engine = None

    def run(self, state: AgentState) -> AgentState:

        print("\n==============================")
        print("  VERIFICATION AGENT")
        print("==============================")

        if self.clip_engine is None:
            self.clip_engine = CLIPEngine.shared()

        image_path = (
            state.detection.processed_image_path
            or state.detection.image_path
        )

        verified = self._filter_detections(
            state,
            image_path,
        )

        state.verification.verified_objects = verified

        if verified:
            state.verification.confidence_score = mean(
                item["confidence"]
                for item in verified
            )
        else:
            state.verification.confidence_score = 0.0

        print()
        print(f"Verified objects: {len(verified)}")
        print(
            "Verification confidence: "
            f"{state.verification.confidence_score:.3f}"
        )

        return state

    def _filter_detections(
        self,
        state: AgentState,
        image_path: str | None,
    ) -> list[dict]:

        if not image_path:
            return []

        if not os.path.exists(image_path):
            print(
                f"Verification image does not exist: "
                f"{image_path}"
            )
            return []

        image = Image.open(image_path).convert("RGB")

        target = canonicalize_target(
            state.query.target
        )

        attributes = state.query.attributes

        requested_size = (
            attributes.get("size", "")
            .strip()
            .lower()
        )

        quantity = (
            state.query.quantity
            or attributes.get("quantity", "all")
        )

        prompts = build_verification_prompts(
            target,
            attributes,
        )

        if not prompts:
            return []

        requested_prompt = prompts[0]

        detections = []
        crops = []

        # ----------------------------------
        # Stage 1 — geometric filtering
        # ----------------------------------

        for item in state.detection.objects_found:

            detection = Detection(
                label=item["label"],
                confidence=float(
                    item["confidence"]
                ),
                bbox=[
                    float(value)
                    for value in item["bbox"]
                ],
            )

            if (
                detection.confidence
                < state.strategy.confidence_threshold
            ):
                continue

            if (
                canonicalize_target(detection.label)
                != target
            ):
                continue

            if (
                requested_size
                and not self._matches_size(
                    detection.bbox,
                    image.size,
                    requested_size,
                )
            ):
                continue

            crop = self._crop_image_with_context(
                image,
                detection.bbox,
            )

            if crop is None:
                continue

            detections.append(detection)
            crops.append(crop)

        if not detections:
            return []

        # ----------------------------------
        # Stage 2 — batched CLIP inference
        # ----------------------------------

        score_results = (
            self.clip_engine
            .score_images_against_texts(
                crops,
                prompts,
            )
        )

        candidates = []

        has_attributes = bool(
            attributes.get("color")
            or attributes.get("size")
        )

        for detection, scores in zip(
            detections,
            score_results,
        ):

            if not scores:
                continue

            requested_score = scores.get(
                requested_prompt,
                -1.0,
            )

            alternatives = [
                score
                for prompt, score in scores.items()
                if prompt != requested_prompt
            ]

            best_alternative = (
                max(alternatives)
                if alternatives
                else -1.0
            )

            margin = (
                requested_score
                - best_alternative
            )

            selected_prompt, selected_score = max(
                scores.items(),
                key=lambda item: item[1],
            )

            print()
            print(
                f"Candidate: {detection.label} "
                f"{detection.confidence:.3f}"
            )

            print(
                f"Requested: {requested_prompt} "
                f"{requested_score:.3f}"
            )

            print(
                f"Best match: {selected_prompt} "
                f"{selected_score:.3f}"
            )

            print(
                f"Attribute margin: {margin:.3f}"
            )

            # ----------------------------------
            # CLIP acceptance
            # ----------------------------------

            if (
                requested_score
                < self.MIN_CLIP_SIMILARITY
            ):
                print(
                    "Rejected: CLIP similarity "
                    "too low"
                )
                continue

            if has_attributes:

                if selected_prompt != requested_prompt:
                    print(
                        "Rejected: another attribute "
                        "prompt matched better"
                    )
                    continue

                if (
                    margin
                    < self.MIN_ATTRIBUTE_MARGIN
                ):
                    print(
                        "Rejected: attribute evidence "
                        "too weak"
                    )
                    continue

            # ----------------------------------
            # Combined confidence
            # ----------------------------------

            clip_quality = max(
                0.0,
                min(
                    1.0,
                    (requested_score + 1.0) / 2.0,
                ),
            )

            combined_confidence = (
                0.70 * detection.confidence
                + 0.30 * clip_quality
            )

            print(
                "Accepted: "
                f"combined={combined_confidence:.3f}"
            )

            candidates.append(
                {
                    "label": detection.label,
                    "confidence": float(
                        combined_confidence
                    ),
                    "detector_confidence": float(
                        detection.confidence
                    ),
                    "clip_similarity": float(
                        requested_score
                    ),
                    "clip_margin": float(margin),
                    "bbox": detection.bbox,
                }
            )

        # ----------------------------------
        # Quantity handling
        # ----------------------------------

        candidates.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        if quantity == "one":
            return candidates[:1]

        return candidates

    @staticmethod
    def _matches_size(
        bbox: list[float],
        image_size: tuple[int, int],
        requested_size: str,
    ) -> bool:

        image_width, image_height = image_size

        box_width = max(
            1.0,
            bbox[2] - bbox[0],
        )

        box_height = max(
            1.0,
            bbox[3] - bbox[1],
        )

        relative_area = (
            box_width * box_height
        ) / float(
            image_width * image_height
        )

        if requested_size in {
            "small",
            "tiny",
            "little",
        }:
            return relative_area <= 0.08

        if requested_size == "medium":
            return (
                0.08
                < relative_area
                < 0.20
            )

        if requested_size in {
            "large",
            "big",
        }:
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

        x1 = int(
            max(
                0,
                x1 - pad_x,
            )
        )

        y1 = int(
            max(
                0,
                y1 - pad_y,
            )
        )

        x2 = int(
            min(
                image.width,
                x2 + pad_x,
            )
        )

        y2 = int(
            min(
                image.height,
                y2 + pad_y,
            )
        )

        if x2 <= x1 or y2 <= y1:
            return None

        return image.crop(
            (
                x1,
                y1,
                x2,
                y2,
            )
        )