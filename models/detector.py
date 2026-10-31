from PIL import Image
import re

from models.schemas import DetectionResult
from models.yolo_world import YOLOWorldDetector
from models.enhanced_postprocessor import EnhancedPostProcessor
from models.detection_quality_optimizer import DetectionQualityOptimizer
from models.detection_config import DetectionConfig, DEFAULT_CONFIG
from utils.search_utils import canonicalize_target

class DetectionEngine:

    def __init__(self, model_path: str | None = None, config: DetectionConfig | None = None):

        self.detector = YOLOWorldDetector(model_path=model_path)
        self.config = config if config is not None else DEFAULT_CONFIG

    def detect(
        self,
        image_path: str,
        target: str,
        confidence: float | None = None,
        config: DetectionConfig | None = None,
    ):

        print("\n==============================")
        print("   DETECTION ENGINE")
        print("==============================")

        # Use provided config or instance config
        use_config = config if config is not None else self.config
        
        # Use provided confidence or config default
        use_confidence = confidence if confidence is not None else use_config.base_confidence_threshold

        raw_detections, inference_time = self.detector.detect(
            image_path=image_path,
            classes=[target],
            confidence=use_confidence,
        )

        raw_count = len(raw_detections)

        if target == "":
            target_matched = raw_detections
        else:
            requested_target = canonicalize_target(target)
            target_matched = [
                d for d in raw_detections
                if canonicalize_target(d.label) == requested_target
            ]

        # Use enhanced post-processor
        img = Image.open(image_path)
        image_width, image_height = img.size

        filtered_detections = EnhancedPostProcessor.apply_nms(
            target_matched,
            config=use_config,
            image_width=image_width,
            image_height=image_height,
        )

        # Apply quality optimization
        filtered_detections, opt_report = DetectionQualityOptimizer.optimize_detections(
            filtered_detections,
            image_path,
            config=use_config,
            verbose=False,
        )

        print(f"\nRaw detections : {raw_count}")
        print(f"Target matches : {len(target_matched)}")
        print(f"After NMS      : {len(filtered_detections)}")

        return DetectionResult(
            model_name="YOLO-World",
            inference_time=inference_time,
            image_width=image_width,
            image_height=image_height,
            raw_detections=raw_detections,
            filtered_detections=filtered_detections,
        )

    @staticmethod
    def _normalize_label(label: str) -> str:

        normalized = label.strip().lower()
        normalized = normalized.replace("_", " ").replace("-", " ")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized