from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.postprocess import set_postprocess_backend
set_postprocess_backend("numpy")

from models.postprocessor import DetectionPostProcessor
from models.enhanced_postprocessor import EnhancedPostProcessor
from models.schemas import Detection
from models.detection_config import DetectionConfig, DEFAULT_CONFIG
from models.detection_quality_optimizer import DetectionQualityOptimizer
from utils.search_utils import canonicalize_target, normalize_label
from utils.model_paths import resolve_yolo_world_weights, is_visdrone_checkpoint


class SAHIEngine:

    # Keep a broad vocabulary so YOLO-World can preserve the true class label
    # instead of forcing every object into the single user-entered prompt.
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

        if self.model_path in SAHIEngine._MODEL_CACHE:
            self.model = SAHIEngine._MODEL_CACHE[self.model_path]
            self.is_visdrone = is_visdrone_checkpoint(getattr(self.model, "model", None)) or is_visdrone_checkpoint(self.model_path)
            return

        print("\nLoading SAHI Engine...")

        import torch
        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=0.25,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.is_visdrone = is_visdrone_checkpoint(getattr(self.model, "model", None)) or is_visdrone_checkpoint(self.model_path)
        SAHIEngine._MODEL_CACHE[self.model_path] = self.model

        if self.is_visdrone:
            print("SAHI Engine configured for fine-tuned VisDrone detector.")
        else:
            print("SAHI Engine configured for open-vocabulary YOLO-World.")

        print("SAHI Engine Loaded Successfully!")

    def detect(
        self,
        image_path: str,
        target: str,
        config: DetectionConfig | None = None,
    ):

        if config is None:
            config = DEFAULT_CONFIG

        requested_target = canonicalize_target(target)

        # For open-vocabulary YOLO-World, set vocabulary dynamically.
        # For fine-tuned VisDrone checkpoints, do NOT overwrite the 10-class head.
        if not self.is_visdrone and hasattr(self.model.model, "set_classes"):
            self.model.model.set_classes(self._build_vocabulary(requested_target))

        self.model.confidence_threshold = config.base_confidence_threshold

        slice_height, slice_width, overlap_height_ratio, overlap_width_ratio = self._get_slice_config(
            image_path,
            small_object_focus=config.enable_small_object_detection
        )

        result = get_sliced_prediction(
            image=image_path,
            detection_model=self.model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_type="NMS",
        )

        detections: list[Detection] = []

        for obj in result.object_prediction_list:

            predicted_label = canonicalize_target(obj.category.name)

            # Filter strictly by the actual model prediction, not the user input.
            if requested_target != "" and predicted_label != requested_target:
                continue

            detections.append(
                Detection(
                    label=predicted_label,
                    confidence=float(obj.score.value),
                    bbox=[
                        float(obj.bbox.minx),
                        float(obj.bbox.miny),
                        float(obj.bbox.maxx),
                        float(obj.bbox.maxy),
                    ],
                )
            )

        # Use enhanced post-processor with configuration
        from PIL import Image
        img = Image.open(image_path)
        image_width, image_height = img.size

        deduped = EnhancedPostProcessor.apply_nms(
            detections,
            config=config,
            image_width=image_width,
            image_height=image_height,
        )

        if len(deduped) != len(detections):
            print(f"Merged duplicate slice detections: {len(detections)} -> {len(deduped)}")

        return [
            {
                "class": detection.label,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
            }
            for detection in deduped
        ]

    def _build_vocabulary(self, requested_target: str) -> list[str]:
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

        vocabulary = list(self.DEFAULT_CLASSES)

        if requested_target not in vocabulary:
            vocabulary.append(requested_target)
        
        # Add synonyms if they exist
        target_lower = requested_target.lower()
        if target_lower in SYNONYMS:
            for synonym in SYNONYMS[target_lower]:
                if synonym not in vocabulary:
                    vocabulary.append(synonym)

        return vocabulary

    def _get_slice_config(self, image_path: str, small_object_focus: bool = False) -> tuple[int, int, float, float]:

        from PIL import Image

        width, height = Image.open(image_path).size
        
        # Use configuration system
        config = DEFAULT_CONFIG
        return config.get_slice_config(width, height, small_object_focus=small_object_focus)