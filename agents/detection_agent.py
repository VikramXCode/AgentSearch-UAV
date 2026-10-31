from dataclasses import dataclass

from models.detector import DetectionEngine
from models.sahi_engine import SAHIEngine
from models.schemas import Detection
from models.detection_config import DetectionConfig, DEFAULT_CONFIG
from models.detection_quality_optimizer import DetectionQualityOptimizer

from utils.visualizer import DetectionVisualizer
from utils.paths import DETECTION_OUTPUT_PATH

from workflows.state import AgentState


import time


@dataclass
class DetectionRunResult:
    detections: list[Detection]
    output_image_path: str
    processed_image_path: str
    detector_name: str
    detection_time: float = 0.0


class DetectionAgent:

    def __init__(self, model_path: str | None = None):
        from utils.model_paths import DEFAULT_YOLO_WORLD_WEIGHTS, PROJECT_ROOT
        from pathlib import Path
        self.e3_detector = DetectionEngine(model_path=str(DEFAULT_YOLO_WORLD_WEIGHTS))
        
        ov_path = PROJECT_ROOT / "weights" / "yolov8l-worldv2.pt"
        if Path(ov_path).exists():
            self.ov_detector = DetectionEngine(model_path=str(ov_path))
        else:
            self.ov_detector = None
            
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

        state.detection.detection_time = run_result.detection_time
        state.strategy.detector = run_result.detector_name

        print(
            f"Raw mission detections: {len(state.detection.objects_found)} "
            f"(in {run_result.detection_time:.3f}s)"
        )

        return state

    def _detect(
        self,
        image_path: str,
        state: AgentState,
    ) -> DetectionRunResult:

        target = state.query.target

        # Create config from strategy state
        config = self._create_config_from_strategy(state.strategy)

        # ==============================================
        # SAHI + YOLO-WORLD
        # ==============================================

        if state.strategy.enable_sahi:

            t_start = time.perf_counter()
            detections = self.sahi_detector.detect(
                image_path,
                target,
                config=config,
            )
            detection_time = time.perf_counter() - t_start

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
            active_detector = self.ov_detector if state.strategy.detector == "YOLO-World-OV" else self.e3_detector
            
            if active_detector is None:
                raise RuntimeError(
                    "Open-Vocabulary detector requested but 'yolov8l-worldv2.pt' is not downloaded. "
                    "Please download it manually into the weights/ directory."
                )
                
            t_start = time.perf_counter()
            result = active_detector.detect(
                image_path=image_path,
                target=target,
                confidence=(
                    state.strategy.confidence_threshold
                ),
                config=config,
            )
            detection_time = result.inference_time if result.inference_time > 0 else (time.perf_counter() - t_start)

            normalized_detections = list(
                result.filtered_detections
            )

            detector_name = "YOLO-World-OV" if state.strategy.detector == "YOLO-World-OV" else result.model_name

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
            detection_time=detection_time,
        )

    def _create_config_from_strategy(self, strategy_state) -> DetectionConfig:
        """Create detection config from strategy state."""
        config = DEFAULT_CONFIG
        config.base_confidence_threshold = strategy_state.confidence_threshold
        return config

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