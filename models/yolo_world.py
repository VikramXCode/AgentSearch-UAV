from ultralytics import YOLO
import time

from models.schemas import Detection
from models.base_detector import BaseDetector


class YOLOWorldDetector(BaseDetector):

    def __init__(self, model_path: str = "weights/yolov8s-world.pt"):

        self.model_path = model_path
        self.model = None

        self.load_model()

    def load_model(self):

        print("\nLoading YOLO-World...")

        self.model = YOLO(self.model_path)

        print("YOLO-World Loaded Successfully!")

    def detect(
        self,
        image_path: str,
        classes: list[str],
        confidence: float = 0.25,
    ):

        # Set the vocabulary
        self.model.set_classes(classes)

        # Measure complete inference time
        start_time = time.perf_counter()

        results = self.model.predict(
            source=image_path,
            conf=confidence,
            verbose=False,
        )

        inference_time = time.perf_counter() - start_time

        detections = []

        print("\n==============================")
        print("YOLO-WORLD RAW OUTPUT")
        print("==============================")

        for result in results:

            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            print(f"\nNumber of boxes: {len(boxes)}")

            for i, box in enumerate(boxes, start=1):

                label = result.names[int(box.cls)]
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()

                print(f"\nDetection {i}")
                print(f"Label      : {label}")
                print(f"Confidence : {conf:.3f}")
                print(f"BBox       : {bbox}")

                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        bbox=bbox,
                    )
                )

        return detections, inference_time