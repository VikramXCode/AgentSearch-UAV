from PIL import Image

from models.schemas import DetectionResult
from models.yolo_world import YOLOWorldDetector
from models.postprocessor import DetectionPostProcessor

class DetectionEngine:

    def __init__(self):

        self.detector = YOLOWorldDetector()
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

        filtered_detections = DetectionPostProcessor.apply_nms(
            raw_detections
        )

        print(f"\nRaw detections : {raw_count}")
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