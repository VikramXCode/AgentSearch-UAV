from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from models.postprocessor import DetectionPostProcessor
from models.schemas import Detection
from utils.search_utils import canonicalize_target, normalize_label
from utils.model_paths import resolve_yolo_world_weights


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

    def __init__(self, model_path: str | None = None):

        print("\nLoading SAHI Engine...")

        self.model_path = resolve_yolo_world_weights(model_path)

        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=0.10,
            device="cpu"
        )

        print("SAHI Engine Loaded Successfully!")

    def detect(
        self,
        image_path: str,
        target: str,
    ):

        requested_target = canonicalize_target(target)

        # Set a broad vocabulary, then filter the predictions down to the
        # requested class after inference. This avoids the one-class coercion
        # bug that can relabel unrelated objects as the user prompt.
        self.model.model.set_classes(self._build_vocabulary(requested_target))

        slice_height, slice_width, overlap_height_ratio, overlap_width_ratio = self._get_slice_config(image_path)

        result = get_sliced_prediction(
            image=image_path,
            detection_model=self.model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )

        detections: list[Detection] = []

        for obj in result.object_prediction_list:

            predicted_label = canonicalize_target(obj.category.name)

            # Filter strictly by the actual model prediction, not the user input.
            if predicted_label != requested_target:
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

        deduped = DetectionPostProcessor.apply_nms(detections, iou_threshold=0.45)

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

        vocabulary = list(self.DEFAULT_CLASSES)

        if requested_target not in vocabulary:
            vocabulary.append(requested_target)

        return vocabulary

    def _get_slice_config(self, image_path: str) -> tuple[int, int, float, float]:

        from PIL import Image

        width, height = Image.open(image_path).size
        max_side = max(width, height)
        area = width * height

        # Larger UAV images benefit from bigger slices and slightly smaller overlap
        # so the detector stays efficient without losing coverage.
        if area >= 4_000_000 or max_side >= 2400:
            return 640, 640, 0.20, 0.20

        if area >= 1_500_000 or max_side >= 1600:
            return 512, 512, 0.20, 0.20

        return 384, 384, 0.25, 0.25