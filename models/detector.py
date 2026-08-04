from PIL import Image
import re

from models.schemas import DetectionResult
from models.yolo_world import YOLOWorldDetector
from models.postprocessor import DetectionPostProcessor
from utils.search_utils import canonicalize_target

class DetectionEngine:

    def __init__(self, model_path: str | None = None):

        self.detector = YOLOWorldDetector(model_path=model_path)
    def detect(
        self,
        image_path: str,
        target: str,
        confidence: float = 0.25,
    ):

        print("\n==============================")
        print("   DETECTION ENGINE")
        print("==============================")

        raw_detections, inference_time = self.detector.detect(
            image_path=image_path,
            classes=[target],
            confidence=confidence,
        )

        raw_count = len(raw_detections)

        requested_target = canonicalize_target(target)

        target_matched = [
            d for d in raw_detections
            if canonicalize_target(d.label) == requested_target
        ]

        filtered_detections = DetectionPostProcessor.apply_nms(
            target_matched
        )

        print(f"\nRaw detections : {raw_count}")
        print(f"Target matches : {len(target_matched)}")
        print(f"After NMS      : {len(filtered_detections)}")

        width, height = Image.open(image_path).size

        return DetectionResult(
            model_name="YOLO-World",
            inference_time=inference_time,
            image_width=width,
            image_height=height,
            raw_detections=raw_detections,
            filtered_detections=filtered_detections,
        )

    @staticmethod
    def _normalize_label(label: str) -> str:

        normalized = label.strip().lower()
        normalized = normalized.replace("_", " ").replace("-", " ")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized