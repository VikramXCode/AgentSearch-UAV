import os
from statistics import mean

from PIL import Image

from models.clip_engine import CLIPEngine
from models.schemas import Detection
from utils.search_utils import build_verification_prompts, canonicalize_target
from workflows.state import AgentState


class VerificationAgent:

    def __init__(self):
        # Load CLIP lazily so the pipeline can start even when verification is not needed.
        self.clip_engine = None

    def run(self, state: AgentState) -> AgentState:

        print("\n==============================")
        print("  VERIFICATION AGENT")
        print("==============================")

        if self.clip_engine is None:
            self.clip_engine = CLIPEngine.shared()

        image_path = state.detection.processed_image_path or state.detection.image_path
        verified = self._filter_detections(state, image_path)

        state.verification.verified_objects = [
            {
                "label": detection.label,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
            }
            for detection in verified
        ]
        state.verification.confidence_score = mean([detection.confidence for detection in verified]) if verified else 0.0

        print(f"Verified objects: {len(verified)}")
        print(f"Verification confidence: {state.verification.confidence_score:.3f}")

        return state

    def _filter_detections(self, state: AgentState, image_path: str | None) -> list[Detection]:

        if not image_path:
            image_path = state.detection.image_path

        if not image_path or not os.path.exists(image_path):
            return []

        target = canonicalize_target(state.query.target)
        requested_size = state.query.attributes.get("size", "").strip().lower()
        quantity = state.query.quantity or state.query.attributes.get("quantity", "all")

        image = Image.open(image_path).convert("RGB")
        prompt_candidates = build_verification_prompts(state.query.target, state.query.attributes)
        requested_prompt = prompt_candidates[0] if prompt_candidates else target

        candidates = []

        for object_item in state.detection.objects_found:
            detection = Detection(
                label=object_item["label"],
                confidence=float(object_item["confidence"]),
                bbox=[float(value) for value in object_item["bbox"]],
            )

            if float(detection.confidence) < state.strategy.confidence_threshold:
                continue

            if canonicalize_target(detection.label) != target:
                continue

            if requested_size and not self._matches_size(detection.bbox, image.size, requested_size):
                continue

            crop = self._crop_image(image, detection.bbox)
            if crop is None:
                continue

            scores = self.clip_engine.score_image_against_texts(crop, prompt_candidates)

            if len(scores) == 0:
                continue

            print(f"\nDetected: {detection.label}")
            print("Similarity:")

            for prompt, similarity in sorted(scores.items(), key=lambda item: item[1], reverse=True):
                print(f"  {prompt} = {similarity:.2f}")

            selected_prompt, selected_score = max(scores.items(), key=lambda item: item[1])
            print(f"Selected: {selected_prompt} ({selected_score:.2f})")

            if selected_prompt != requested_prompt:
                print("Accepted: no")
                continue

            print("Accepted: yes")

            candidates.append(
                {
                    "detection": Detection(
                        label=detection.label,
                        confidence=detection.confidence,
                        bbox=detection.bbox,
                    ),
                    "clip_similarity": float(selected_score),
                    "selected_prompt": selected_prompt,
                }
            )

        if quantity == "one" and len(candidates) > 1:
            candidates = [max(candidates, key=lambda item: (item["clip_similarity"], item["detection"].confidence))]

        return [candidate["detection"] for candidate in candidates]

    @staticmethod
    def _matches_size(bbox: list[float], image_size: tuple[int, int], requested_size: str) -> bool:

        image_width, image_height = image_size
        box_width = max(1.0, bbox[2] - bbox[0])
        box_height = max(1.0, bbox[3] - bbox[1])
        relative_area = (box_width * box_height) / float(image_width * image_height)

        if requested_size == "small":
            return relative_area <= 0.08
        if requested_size == "medium":
            return 0.08 < relative_area < 0.20
        if requested_size == "large":
            return relative_area >= 0.20

        return True

    @staticmethod
    def _crop_image(image: Image.Image, bbox: list[float]) -> Image.Image | None:

        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1 = max(0, min(x1, image.width - 1))
        y1 = max(0, min(y1, image.height - 1))
        x2 = max(x1 + 1, min(x2, image.width))
        y2 = max(y1 + 1, min(y2, image.height))

        if x2 <= x1 or y2 <= y1:
            return None

        return image.crop((x1, y1, x2, y2))
