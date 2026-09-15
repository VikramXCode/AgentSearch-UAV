"""
Post-detection verification and filtering to reduce false positives.
"""

from models.schemas import Detection
from models.color_verifier import ColorVerifier
from typing import Optional


class DetectionVerifier:
    """Verify and filter detections based on physical properties."""

    @staticmethod
    def apply_size_filter(
        detections: list[Detection],
        image_width: int,
        image_height: int,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.95,
    ) -> list[Detection]:
        """
        Filter detections by bounding box area relative to image size.
        
        Args:
            detections: List of detections
            image_width: Image width in pixels
            image_height: Image height in pixels
            min_area_ratio: Minimum detection area as fraction of image (default 0.1%)
            max_area_ratio: Maximum detection area as fraction of image (default 95%)
        
        Returns:
            Filtered detections
        """
        
        image_area = image_width * image_height
        min_area = image_area * min_area_ratio
        max_area = image_area * max_area_ratio
        
        filtered = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            
            if min_area <= bbox_area <= max_area:
                filtered.append(detection)
            else:
                status = "too small" if bbox_area < min_area else "too large"
                print(f"  Filtering out: {detection.label} ({status}, area={bbox_area:.0f})")
        
        return filtered

    @staticmethod
    def apply_aspect_ratio_filter(
        detections: list[Detection],
        min_ratio: float = 0.2,
        max_ratio: float = 5.0,
    ) -> list[Detection]:
        """
        Filter detections by aspect ratio to remove elongated/thin detections.
        
        Args:
            detections: List of detections
            min_ratio: Minimum width/height ratio
            max_ratio: Maximum width/height ratio
        
        Returns:
            Filtered detections
        """
        
        filtered = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            width = x2 - x1
            height = y2 - y1
            
            if height == 0:
                continue
            
            aspect_ratio = width / height
            
            if min_ratio <= aspect_ratio <= max_ratio:
                filtered.append(detection)
            else:
                print(f"  Filtering out: {detection.label} (bad aspect ratio={aspect_ratio:.2f})")
        
        return filtered

    @staticmethod
    def apply_edge_proximity_filter(
        detections: list[Detection],
        image_width: int,
        image_height: int,
        edge_margin: float = 0.05,
    ) -> list[Detection]:
        """
        Filter detections that are too close to image edges (likely partial objects).
        
        Args:
            detections: List of detections
            image_width: Image width in pixels
            image_height: Image height in pixels
            edge_margin: Fraction of image to consider as edge (default 5%)
        
        Returns:
            Filtered detections
        """
        
        margin_x = image_width * edge_margin
        margin_y = image_height * edge_margin
        
        filtered = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Check if detection is too close to edges
            if (x1 < margin_x or x2 > image_width - margin_x or
                y1 < margin_y or y2 > image_height - margin_y):
                print(f"  Filtering out: {detection.label} (too close to edge)")
                continue
            
            filtered.append(detection)
        
        return filtered

    @staticmethod
    def apply_confidence_filter(
        detections: list[Detection],
        min_confidence: float = 0.60,
    ) -> list[Detection]:
        """
        Filter detections below confidence threshold.
        
        Args:
            detections: List of detections
            min_confidence: Minimum confidence score
        
        Returns:
            Filtered detections
        """
        
        filtered = []
        
        for detection in detections:
            if detection.confidence >= min_confidence:
                filtered.append(detection)
            else:
                print(f"  Filtering out: {detection.label} (low confidence={detection.confidence:.3f})")
        
        return filtered

    @staticmethod
    def apply_all_filters(
        detections: list[Detection],
        image_width: int,
        image_height: int,
        min_confidence: float = 0.60,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.95,
        min_aspect_ratio: float = 0.2,
        max_aspect_ratio: float = 5.0,
        check_edge_proximity: bool = True,
    ) -> list[Detection]:
        """
        Apply all verification filters in sequence.
        
        Args:
            detections: List of detections
            image_width: Image width
            image_height: Image height
            min_confidence: Minimum confidence
            min_area_ratio: Minimum area as fraction of image
            max_area_ratio: Maximum area as fraction of image
            min_aspect_ratio: Minimum width/height ratio
            max_aspect_ratio: Maximum width/height ratio
            check_edge_proximity: Whether to filter edge-adjacent detections
        
        Returns:
            Filtered detections
        """
        
        print(f"\nApplying post-detection verification filters...")
        print(f"  Input detections: {len(detections)}")
        
        # Apply filters in order
        detections = DetectionVerifier.apply_confidence_filter(
            detections, min_confidence
        )
        print(f"  After confidence filter: {len(detections)}")
        
        detections = DetectionVerifier.apply_size_filter(
            detections, image_width, image_height, min_area_ratio, max_area_ratio
        )
        print(f"  After size filter: {len(detections)}")
        
        detections = DetectionVerifier.apply_aspect_ratio_filter(
            detections, min_aspect_ratio, max_aspect_ratio
        )
        print(f"  After aspect ratio filter: {len(detections)}")
        
        if check_edge_proximity:
            detections = DetectionVerifier.apply_edge_proximity_filter(
                detections, image_width, image_height
            )
            print(f"  After edge proximity filter: {len(detections)}")
        
        return detections


class ContextualVerifier:
    """Use contextual knowledge to verify detections."""

    @staticmethod
    def verify_with_context(
        detections: list[Detection],
        context: dict,
    ) -> list[Detection]:
        """
        Filter detections based on context knowledge.
        
        Example context:
        {
            "min_objects": 1,
            "max_objects": 10,
            "expected_location": "center",  # "center", "anywhere", "edges"
            "environment": "urban",  # Helps predict likely objects
        }
        """
        
        if not context:
            return detections
        
        filtered = detections
        
        # Check count constraints
        min_obj = context.get("min_objects", 0)
        max_obj = context.get("max_objects", float('inf'))
        
        if len(filtered) < min_obj:
            print(f"Warning: Expected at least {min_obj} objects, found {len(filtered)}")
        elif len(filtered) > max_obj:
            # Keep top-confidence detections
            filtered = sorted(filtered, key=lambda d: d.confidence, reverse=True)[:max_obj]
            print(f"Keeping top {max_obj} detections by confidence")
        
        return filtered
