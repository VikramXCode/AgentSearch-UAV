from dataclasses import dataclass

from models.detector import DetectionEngine
from models.sahi_engine import SAHIEngine
from models.schemas import Detection

from utils.visualizer import DetectionVisualizer
from utils.paths import DETECTION_OUTPUT_PATH

from workflows.state import AgentState


@dataclass
class DetectionRunResult:
    detections: list[Detection]
    output_image_path: str
    processed_image_path: str
    detector_name: str


class DetectionAgent:

    def __init__(self, model_path: str | None = None):
        self.yolo_detector = DetectionEngine(model_path=model_path)
        self.sahi_detector = SAHIEngine(model_path=model_path)

    def run(
        self,
        state: AgentState,
        image_path: str,
    ) -> AgentState:

        print("\n==============================")
        print("    DETECTION AGENT")
        print("==============================")

        # image_path is already processed by ToolAgent
        run_result = self._detect(
            image_path=image_path,
            state=state,
        )

        state.detection.objects_found = [
            {
                "label": detection.label,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
            }
            for detection in run_result.detections
        ]

        # Do NOT overwrite original image_path.
        # ToolAgent already stores:
        # state.detection.image_path = original image
        # state.detection.processed_image_path = processed image

        state.detection.processed_image_path = (
            run_result.processed_image_path
        )

        state.detection.output_image_path = (
            run_result.output_image_path
        )

        state.strategy.detector = run_result.detector_name

        print(
            f"Raw mission detections: "
            f"{len(state.detection.objects_found)}"
        )

        return state

    def _detect(
        self,
        image_path: str,
        state: AgentState,
    ) -> DetectionRunResult:

        target = state.query.target

        # ==============================================
        # SAHI + YOLO-WORLD
        # ==============================================

        if state.strategy.enable_sahi:

            detections = self.sahi_detector.detect(
                image_path,
                target,
            )

            normalized_detections = [
                Detection(
                    label=detection["class"],
                    confidence=float(
                        detection["confidence"]
                    ),
                    bbox=[
                        float(value)
                        for value in detection["bbox"]
                    ],
                )
                for detection in detections
            ]

            detector_name = "YOLO-World + SAHI"

        # ==============================================
        # STANDARD YOLO-WORLD
        # ==============================================

        else:

            result = self.yolo_detector.detect(
                image_path=image_path,
                target=target,
                confidence=(
                    state.strategy.confidence_threshold
                ),
            )

            normalized_detections = list(
                result.filtered_detections
            )

            detector_name = result.model_name

        # ==============================================
        # ANNOTATE
        # ==============================================

        self.save_annotated_image(
            image_path=image_path,
            detections=normalized_detections,
            output_path=DETECTION_OUTPUT_PATH,
        )

        return DetectionRunResult(
            detections=normalized_detections,
            output_image_path=DETECTION_OUTPUT_PATH,
            processed_image_path=image_path,
            detector_name=detector_name,
        )

    def save_annotated_image(
        self,
        image_path: str,
        detections: list,
        output_path: str = DETECTION_OUTPUT_PATH,
    ) -> None:

        normalized_detections = []

        for detection in detections:

            if isinstance(detection, Detection):

                normalized_detections.append(
                    detection
                )

            else:

                normalized_detections.append(
                    Detection(
                        label=detection["label"],
                        confidence=float(
                            detection["confidence"]
                        ),
                        bbox=[
                            float(value)
                            for value
                            in detection["bbox"]
                        ],
                    )
                )

        DetectionVisualizer.draw(
            image_path=image_path,
            detections=normalized_detections,
            output_path=output_path,
        )