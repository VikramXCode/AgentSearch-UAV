from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class SAHIEngine:

    def __init__(self):

        print("\nLoading SAHI Engine...")

        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path="weights/yolov8s-world.pt",
            confidence_threshold=0.10,
            device="cpu"
        )

        print("SAHI Engine Loaded Successfully!")

    def detect(
        self,
        image_path: str,
        target: str,
    ):

        # Set YOLO-World classes
        self.model.model.set_classes([target])

        result = get_sliced_prediction(
            image=image_path,
            detection_model=self.model,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.25,
            overlap_width_ratio=0.25,
        )

        detections = []

        for obj in result.object_prediction_list:

            detections.append(
                {
                    "class": obj.category.name,
                    "confidence": obj.score.value,
                    "bbox": [
                        obj.bbox.minx,
                        obj.bbox.miny,
                        obj.bbox.maxx,
                        obj.bbox.maxy,
                    ],
                }
            )

        return detections