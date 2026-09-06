from ultralytics import YOLO
import time

from models.schemas import Detection
from models.base_detector import BaseDetector
from utils.model_paths import resolve_yolo_world_weights, is_visdrone_checkpoint, VISDRONE_CLASS_NAMES
from utils.search_utils import get_visdrone_classes_for_target, canonicalize_target


class YOLOWorldDetector(BaseDetector):

    DEFAULT_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    _MODEL_CACHE: dict = {}

    def __init__(self, model_path: str | None = None):

        self.model_path = resolve_yolo_world_weights(model_path)
        self.model = None
        self.is_visdrone = False

        self.load_model()

    def load_model(self):
        if self.model_path in YOLOWorldDetector._MODEL_CACHE:
            self.model = YOLOWorldDetector._MODEL_CACHE[self.model_path]
            self.is_visdrone = is_visdrone_checkpoint(self.model) or is_visdrone_checkpoint(self.model_path)
            return

        print("\nLoading Object Detection Model...")

        self.model = YOLO(self.model_path)
        self.is_visdrone = is_visdrone_checkpoint(self.model) or is_visdrone_checkpoint(self.model_path)
        YOLOWorldDetector._MODEL_CACHE[self.model_path] = self.model

        if self.is_visdrone:
            print("Loaded fine-tuned VisDrone detector (10 aerial UAV classes).")
        else:
            print("Loaded open-vocabulary YOLO-World detector.")

    def detect(
        self,
        image_path: str,
        classes: list[str],
        confidence: float = 0.25,
    ):

        # For open-vocabulary YOLO-World, configure vocabulary dynamically.
        # For fine-tuned VisDrone checkpoints, preserve the trained 10-class head
        # so feature-to-class alignment remains intact.
        if not self.is_visdrone and hasattr(self.model, "set_classes"):
            vocabulary = self._build_vocabulary(classes)
            self.model.set_classes(vocabulary)

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

                class_id = int(box.cls)
                label = self._resolve_label(result.names, class_id)
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

    def _build_vocabulary(self, classes: list[str]) -> list[str]:
        """Build vocabulary with synonyms for better detection."""
        
        # Synonym mapping for common objects
        SYNONYMS = {
            "person": ["human", "pedestrian", "people", "man", "woman"],
            "car": ["automobile", "vehicle", "sedan", "truck"],
            "dog": ["canine", "puppy"],
            "cat": ["feline", "kitten"],
            "bird": ["avian", "eagle", "hawk"],
            "airplane": ["aircraft", "plane", "jet"],
            "boat": ["ship", "vessel", "yacht"],
        }

        requested = [c.strip() for c in classes if c and c.strip()]
        base = list(self.DEFAULT_CLASSES)

        existing = {c.lower() for c in base}

        # Add requested classes and their synonyms
        for cls in requested:
            cls_lower = cls.lower()
            if cls_lower not in existing:
                base.append(cls)
                existing.add(cls_lower)
            
            # Add synonyms if they exist
            if cls_lower in SYNONYMS:
                for synonym in SYNONYMS[cls_lower]:
                    if synonym.lower() not in existing:
                        base.append(synonym)
                        existing.add(synonym.lower())

        return base

    @staticmethod
    def _resolve_label(names, class_id: int) -> str:

        if isinstance(names, dict):
            return str(names.get(class_id, class_id))

        if isinstance(names, list):
            if 0 <= class_id < len(names):
                return str(names[class_id])

        return str(class_id)