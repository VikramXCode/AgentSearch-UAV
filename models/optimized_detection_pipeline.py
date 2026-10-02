#!/usr/bin/env python3
"""
Unified Optimized Detection Pipeline for AgentSearch-UAV.

Integrates:
1. Base Full-Image Detection (YOLO-World / Fine-Tuned VisDrone Checkpoint)
2. Object Size & Scene Density Analysis
3. Adaptive SAHI Sliced Inference
4. Selective ROI Super-Resolution
5. Multi-Source Detection Fusion (Weighted Box Fusion)
6. Class-Specific Calibrated Confidence Filtering
7. Scale-Aware Soft-NMS (Linear / Gaussian)
8. Class Conflict Resolution (car ↔ van, ped ↔ people, bike ↔ motor)
9. False Positive Reduction Layer
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO

from models.schemas import Detection, DetectionResult
from utils.adaptive_sahi import AdaptiveSAHI
from utils.class_conflict_resolver import ClassConflictResolver
from utils.class_threshold_calibration import ClassThresholdCalibrator, DEFAULT_CLASS_THRESHOLDS
from utils.detection_fusion import DetectionFusionEngine
from utils.false_positive_filter import FalsePositiveFilter
from utils.improved_soft_nms import ImprovedSoftNMS
from utils.scale_aware_postprocessing import ScaleAwarePostProcessor
from utils.selective_super_resolution import SelectiveSuperResolution

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


class OptimizedDetectionPipeline:
    """Production-grade unified detection pipeline for UAV aerial search."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path) if model_path else (PROJECT_ROOT / "weights" / "best.pt")
        self.config_path = Path(config_path) if config_path else (PROJECT_ROOT / "configs" / "detection_config.json")
        self.device = device

        # Load configuration
        self.config = self._load_config()

        # Load Model
        print(f"[OptimizedDetectionPipeline] Loading detector: {self.model_path}")
        self.model = YOLO(str(self.model_path))

        # Initialize Modular Subsystems
        self.calibrator = ClassThresholdCalibrator(PROJECT_ROOT / "configs" / "class_thresholds.json")
        self.adaptive_sahi = AdaptiveSAHI(
            min_density_for_sahi=self.config.get("adaptive_sahi", {}).get("min_density_trigger", 5),
            min_small_ratio_for_sahi=self.config.get("adaptive_sahi", {}).get("min_small_ratio_trigger", 0.25),
            default_slice_width=self.config.get("adaptive_sahi", {}).get("default_slice_width", 640),
            default_slice_height=self.config.get("adaptive_sahi", {}).get("default_slice_height", 640),
            default_overlap=self.config.get("adaptive_sahi", {}).get("default_overlap", 0.20),
            dense_slice_width=self.config.get("adaptive_sahi", {}).get("dense_slice_width", 480),
            dense_slice_height=self.config.get("adaptive_sahi", {}).get("dense_slice_height", 480),
            dense_overlap=self.config.get("adaptive_sahi", {}).get("dense_overlap", 0.28),
        )
        self.selective_sr = SelectiveSuperResolution(
            scale=self.config.get("selective_super_resolution", {}).get("scale", 2),
            max_rois_per_frame=self.config.get("selective_super_resolution", {}).get("max_rois_per_frame", 4),
            low_conf_threshold=self.config.get("selective_super_resolution", {}).get("low_conf_trigger", 0.35),
            tiny_area_threshold=self.config.get("selective_super_resolution", {}).get("tiny_area_trigger", 576.0),
            roi_padding_ratio=self.config.get("selective_super_resolution", {}).get("roi_padding_ratio", 0.25),
        )
        self.fusion_engine = DetectionFusionEngine(
            iou_threshold=self.config.get("detection_fusion", {}).get("iou_threshold", 0.55),
            source_weights=self.config.get("detection_fusion", {}).get("source_weights"),
            enable_consensus_boost=self.config.get("detection_fusion", {}).get("enable_consensus_boost", True),
        )
        self.soft_nms = ImprovedSoftNMS(
            method=self.config.get("improved_soft_nms", {}).get("method", "class_adaptive"),
            iou_threshold=self.config.get("improved_soft_nms", {}).get("iou_threshold", 0.32),
            duplicate_cutoff_iou=self.config.get("improved_soft_nms", {}).get("duplicate_cutoff_iou", 0.55),
            score_threshold=self.config.get("improved_soft_nms", {}).get("score_threshold", 0.35),
            sigma=self.config.get("improved_soft_nms", {}).get("sigma", 0.25),
            score_decay=self.config.get("improved_soft_nms", {}).get("score_decay", 0.85),
            use_scale_aware=self.config.get("improved_soft_nms", {}).get("use_scale_aware", True),
            scale_iou_map=self.config.get("improved_soft_nms", {}).get("scale_iou_map"),
        )
        self.scale_postprocessor = ScaleAwarePostProcessor(
            tiny_max_area=self.config.get("scale_aware_postprocessing", {}).get("tiny_max_area", 576),
            small_max_area=self.config.get("scale_aware_postprocessing", {}).get("small_max_area", 1024),
            medium_max_area=self.config.get("scale_aware_postprocessing", {}).get("medium_max_area", 9216),
            tiny_conf_discount=self.config.get("scale_aware_postprocessing", {}).get("tiny_conf_discount", 0.05),
            large_conf_boost=self.config.get("scale_aware_postprocessing", {}).get("large_conf_boost", 0.05),
        )
        self.conflict_resolver = ClassConflictResolver(
            conflict_iou_threshold=self.config.get("class_conflict_resolution", {}).get("conflict_iou_threshold", 0.45),
            enable_geometric_rules=self.config.get("class_conflict_resolution", {}).get("enable_geometric_rules", True),
            enable_overlap_arbitration=self.config.get("class_conflict_resolution", {}).get("enable_overlap_arbitration", True),
        )
        self.fp_filter = FalsePositiveFilter(
            min_width=self.config.get("false_positive_filter", {}).get("min_width", 3.0),
            min_height=self.config.get("false_positive_filter", {}).get("min_height", 3.0),
            min_area=self.config.get("false_positive_filter", {}).get("min_area", 9.0),
            min_aspect_ratio=self.config.get("false_positive_filter", {}).get("min_aspect_ratio", 0.06),
            max_aspect_ratio=self.config.get("false_positive_filter", {}).get("max_aspect_ratio", 16.0),
            edge_margin_ratio=self.config.get("false_positive_filter", {}).get("edge_margin_ratio", 0.002),
        )

        print("[OptimizedDetectionPipeline] Subsystems successfully initialized.")

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[OptimizedDetectionPipeline] Error loading config {self.config_path}: {e}")
        return {}

    def _raw_predict_images(
        self,
        images: List[Image.Image],
        imgsz: int = 960,
        conf: float = 0.08,
    ) -> List[List[Detection]]:
        """Predict a batch of PIL images using the underlying detector."""
        if not images:
            return []

        results = []
        batch_size = 4
        for i in range(0, len(images), batch_size):
            chunk = images[i:i + batch_size]
            chunk_results = self.model.predict(
                source=chunk,
                imgsz=imgsz,
                conf=conf,
                iou=0.45,
                device=self.device,
                verbose=False,
            )
            results.extend(chunk_results)

        batch_detections: List[List[Detection]] = []
        for r in results:
            img_dets: List[Detection] = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)

                for b, s, c in zip(xyxy, confs, classes):
                    if 0 <= c < len(CLASS_NAMES):
                        img_dets.append(Detection(
                            label=CLASS_NAMES[c],
                            confidence=float(s),
                            bbox=[float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                            class_id=int(c),
                            source="full_image",
                        ))
            batch_detections.append(img_dets)

        return batch_detections

    def detect_image(
        self,
        image_input: Union[str, Path, Image.Image],
        target_classes: Optional[List[str]] = None,
        confidence_override: Optional[float] = None,
        enable_sahi: Optional[bool] = None,
        enable_sr: Optional[bool] = None,
    ) -> DetectionResult:
        """
        Execute full multi-stage unified optimization pipeline on a single image.
        """
        t_start = time.perf_counter()
        latencies: Dict[str, float] = {}

        # 1. Load Image
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(str(image_input)).convert("RGB")
        else:
            pil_image = image_input.convert("RGB")
        orig_w, orig_h = pil_image.size

        # 2. Base Full-Image Detection
        t0 = time.perf_counter()
        base_dets = self._raw_predict_images(
            [pil_image],
            imgsz=self.config.get("image_size", 960),
            conf=0.08,
        )[0]
        latencies["base_detection"] = round(time.perf_counter() - t0, 4)

        # 3. Adaptive Difficulty Analysis & Processing
        t0 = time.perf_counter()
        if not hasattr(self, 'adaptive_processor'):
            from utils.adaptive_region_processor import AdaptiveRegionProcessor
            self.adaptive_processor = AdaptiveRegionProcessor()
            
        def _detect_wrapper(patches, conf=0.10, imgsz=640):
            return self._raw_predict_images(patches, imgsz=imgsz, conf=conf)
            
        processed_dets, filtered_base_dets, processing_logs = self.adaptive_processor.process_regions(
            pil_image,
            base_dets,
            detect_fn=_detect_wrapper
        )
        latencies["adaptive_processing"] = round(time.perf_counter() - t0, 4)

        # 3.5 Filter out raw unverified noise
        # Any base detection that wasn't verified by an adaptive branch and has very low confidence is likely noise.
        strong_base_dets = [d for d in filtered_base_dets if d.confidence >= 0.35]

        # 4. Detection Fusion
        t0 = time.perf_counter()
        raw_all = strong_base_dets + processed_dets
        if self.config.get("detection_fusion", {}).get("enabled", True):
            fused_dets = self.fusion_engine.fuse(strong_base_dets, processed_dets, [])
        else:
            fused_dets = raw_all
        latencies["detection_fusion"] = round(time.perf_counter() - t0, 4)

        # 7. Scale-Aware Post-Processing & Class-Specific Confidence Filtering
        t0 = time.perf_counter()
        if confidence_override == "none":
            thresholds_to_use = None
        elif isinstance(confidence_override, (int, float)):
            thresholds_to_use = {c: float(confidence_override) for c in CLASS_NAMES}
            self.soft_nms.score_threshold = float(confidence_override)
        elif isinstance(confidence_override, dict):
            thresholds_to_use = confidence_override
        else:
            thresholds_to_use = self.calibrator.class_thresholds
        scaled_dets = self.scale_postprocessor.process(fused_dets, class_thresholds=thresholds_to_use)

        # 8. Scale-Aware Soft-NMS
        if self.config.get("improved_soft_nms", {}).get("enabled", True):
            nms_dets = self.soft_nms.process_detections(scaled_dets, per_class=True)
        else:
            nms_dets = scaled_dets

        # 9. Class Conflict Resolution
        if self.config.get("class_conflict_resolution", {}).get("enabled", True):
            resolved_dets = self.conflict_resolver.resolve_conflicts(nms_dets)
        else:
            resolved_dets = nms_dets

        # 10. False Positive Reduction Layer
        if self.config.get("false_positive_filter", {}).get("enabled", True):
            final_dets = self.fp_filter.filter_detections(resolved_dets, orig_w, orig_h)
        else:
            final_dets = resolved_dets

        # Filter by requested target classes if specified
        if target_classes:
            targets_norm = {t.strip().lower() for t in target_classes}
            final_dets = [d for d in final_dets if d.label.lower() in targets_norm]

        latencies["post_processing_and_filtering"] = round(time.perf_counter() - t0, 4)
        total_time = round(time.perf_counter() - t_start, 4)

        return DetectionResult(
            model_name="AgentSearch-UAV Optimized Pipeline (v2.5.0)",
            inference_time=total_time,
            image_width=orig_w,
            image_height=orig_h,
            raw_detections=raw_all,
            filtered_detections=final_dets,
            latency_breakdown=latencies,
            pipeline_metadata={
                "adaptive_processing_logs": processing_logs,
                "raw_count": len(raw_all),
                "final_count": len(final_dets),
                "base_count": len(base_dets),
                "processed_count": len(processed_dets),
            },
        )
