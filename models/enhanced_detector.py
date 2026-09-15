"""
Enhanced detection engines with super resolution and ensemble capabilities.
"""

from PIL import Image
import re
import time

from models.schemas import DetectionResult, Detection
from models.yolo_world import YOLOWorldDetector
from models.postprocessor import DetectionPostProcessor
from models.super_resolution import SuperResolutionEngine
from models.clip_engine import CLIPEngine
from utils.search_utils import canonicalize_target


class SuperResolutionDetector:
    """Detector with super resolution preprocessing for better accuracy on small/distant objects."""

    def __init__(self, model_path: str | None = None, scale: int = 2):
        self.detector = YOLOWorldDetector(model_path=model_path)
        self.sr_engine = SuperResolutionEngine()
        self.scale = scale

    def detect(
        self,
        image_path: str,
        target: str,
        confidence: float = 0.60,
        use_sr: bool = True,
    ):
        """
        Detect objects with optional super resolution preprocessing.
        
        Args:
            image_path: Path to input image
            target: Target object to search for
            confidence: Detection confidence threshold
            use_sr: Whether to apply super resolution
        
        Returns:
            DetectionResult with upscaled detections
        """
        
        print("\n==============================")
        print("  SUPER RESOLUTION DETECTOR")
        print("==============================")
        
        # Upscale image if enabled
        if use_sr:
            print(f"\nApplying {self.scale}x super resolution...")
            working_image = self.sr_engine.upscale(
                image_path, 
                scale=self.scale
            )
            print(f"Upscaled image: {working_image}")
        else:
            working_image = image_path

        # Run detection on (upscaled) image
        raw_detections, inference_time = self.detector.detect(
            image_path=working_image,
            classes=[target],
            confidence=confidence,
        )

        raw_count = len(raw_detections)

        requested_target = canonicalize_target(target)

        target_matched = [
            d for d in raw_detections
            if canonicalize_target(d.label) == requested_target
        ]

        filtered_detections = DetectionPostProcessor.apply_nms(
            target_matched
        )

        # Scale bounding boxes back to original if super resolution was used
        if use_sr:
            scale_factor = 1.0 / self.scale
            for detection in filtered_detections:
                detection.bbox = [coord * scale_factor for coord in detection.bbox]

        print(f"\nRaw detections : {raw_count}")
        print(f"Target matches : {len(target_matched)}")
        print(f"After NMS      : {len(filtered_detections)}")

        orig_width, orig_height = Image.open(image_path).size

        return DetectionResult(
            model_name="YOLO-World + SuperResolution",
            inference_time=inference_time,
            image_width=orig_width,
            image_height=orig_height,
            raw_detections=raw_detections,
            filtered_detections=filtered_detections,
        )


class EnsembleDetector:
    """Ensemble detector combining YOLO-World with CLIP verification for higher precision."""

    def __init__(self, model_path: str | None = None):
        self.detector = YOLOWorldDetector(model_path=model_path)
        self.clip_engine = CLIPEngine.shared()

    def detect(
        self,
        image_path: str,
        target: str,
        confidence: float = 0.60,
        clip_threshold: float = 0.25,
    ):
        """
        Detect objects using YOLO-World, then verify with CLIP.
        
        Args:
            image_path: Path to input image
            target: Target object to search for
            confidence: YOLO detection confidence threshold
            clip_threshold: CLIP verification score threshold
        
        Returns:
            DetectionResult with CLIP-verified detections
        """
        
        print("\n==============================")
        print("   ENSEMBLE DETECTOR")
        print("   (YOLO-World + CLIP)")
        print("==============================")

        start_time = time.perf_counter()

        # Run YOLO detection
        raw_detections, yolo_time = self.detector.detect(
            image_path=image_path,
            classes=[target],
            confidence=confidence,
        )

        requested_target = canonicalize_target(target)

        target_matched = [
            d for d in raw_detections
            if canonicalize_target(d.label) == requested_target
        ]

        filtered_detections = DetectionPostProcessor.apply_nms(
            target_matched
        )

        # Verify with CLIP
        print(f"\nVerifying {len(filtered_detections)} detections with CLIP...")
        verified_detections = self._verify_with_clip(
            image_path,
            filtered_detections,
            target,
            clip_threshold,
        )

        inference_time = time.perf_counter() - start_time

        width, height = Image.open(image_path).size

        return DetectionResult(
            model_name="YOLO-World + CLIP Ensemble",
            inference_time=inference_time,
            image_width=width,
            image_height=height,
            raw_detections=raw_detections,
            filtered_detections=verified_detections,
        )

    def _verify_with_clip(
        self,
        image_path: str,
        detections: list[Detection],
        target: str,
        threshold: float,
    ) -> list[Detection]:
        """Verify detections using CLIP image-text similarity."""
        
        image = Image.open(image_path)
        verified = []

        for detection in detections:
            # Crop detection region
            x1, y1, x2, y2 = detection.bbox
            crop = image.crop((x1, y1, x2, y2))

            # Score with CLIP
            prompts = [
                f"a {target}",
                f"this is a {target}",
                target,
                "not a target object",
            ]

            scores = self.clip_engine.score_image_against_texts(crop, prompts)
            
            # Average score for target prompts
            target_score = sum(scores[p] for p in prompts[:3]) / 3.0

            print(f"  {detection.label}: YOLO={detection.confidence:.3f}, CLIP={target_score:.3f}", end="")

            if target_score >= threshold:
                print(" ✓ VERIFIED")
                verified.append(detection)
            else:
                print(" ✗ REJECTED")

        return verified


class AdaptiveDetector:
    """Detector that adapts parameters based on image characteristics."""

    def __init__(self, model_path: str | None = None):
        self.detector = YOLOWorldDetector(model_path=model_path)

    def detect(
        self,
        image_path: str,
        target: str,
    ):
        """
        Detect with adaptive parameters based on image size.
        
        - Small/low-res images: Higher confidence, lower NMS threshold
        - Large/high-res images: Lower confidence, higher NMS threshold
        """
        
        image = Image.open(image_path)
        width, height = image.size
        total_pixels = width * height

        print("\n==============================")
        print("  ADAPTIVE DETECTOR")
        print("==============================")
        print(f"\nImage Size: {width} x {height} ({total_pixels:,} pixels)")

        # Adaptive parameters based on image size
        if total_pixels < 480000:  # < 720p
            confidence = 0.55
            nms_threshold = 0.30
            detection_type = "Low-resolution mode"
        elif total_pixels > 8100000:  # > 4K
            confidence = 0.50
            nms_threshold = 0.40
            detection_type = "High-resolution mode"
        else:  # Standard
            confidence = 0.60
            nms_threshold = 0.35
            detection_type = "Standard mode"

        print(f"Using: {detection_type}")
        print(f"  Confidence threshold: {confidence}")
        print(f"  NMS threshold: {nms_threshold}")

        raw_detections, inference_time = self.detector.detect(
            image_path=image_path,
            classes=[target],
            confidence=confidence,
        )

        requested_target = canonicalize_target(target)

        target_matched = [
            d for d in raw_detections
            if canonicalize_target(d.label) == requested_target
        ]

        filtered_detections = DetectionPostProcessor.apply_nms(
            target_matched,
            iou_threshold=nms_threshold,
        )

        return DetectionResult(
            model_name="Adaptive YOLO-World",
            inference_time=inference_time,
            image_width=width,
            image_height=height,
            raw_detections=raw_detections,
            filtered_detections=filtered_detections,
        )
