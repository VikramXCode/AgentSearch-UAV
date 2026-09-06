"""
Small-object detection optimizer with specialized handling for tiny and distant objects.
Improves detection accuracy for small objects without slowing down overall inference.
"""

from typing import List, Optional, Tuple
from models.schemas import Detection
from models.detection_config import DetectionConfig, DEFAULT_CONFIG


class SmallObjectDetector:
    """Optimizes detection of small/distant objects."""
    
    @staticmethod
    def filter_by_size(
        detections: List[Detection],
        image_width: int,
        image_height: int,
        config: Optional[DetectionConfig] = None,
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        Separate small and large objects.
        
        Returns:
            (small_detections, large_detections) - separated by size
        """
        
        if config is None:
            config = DEFAULT_CONFIG
        
        image_area = image_width * image_height
        
        # Define small object threshold (0.35% of image area)
        small_object_threshold = image_area * 0.0035
        
        small = []
        large = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            area = (x2 - x1) * (y2 - y1)
            
            if area < small_object_threshold:
                small.append(detection)
            else:
                large.append(detection)
        
        return small, large
    
    @staticmethod
    def enhance_small_object_detection(
        detections: List[Detection],
        image_width: int,
        image_height: int,
        config: Optional[DetectionConfig] = None,
    ) -> List[Detection]:
        """
        Apply small-object specific optimizations to improve recall.
        """
        
        if config is None:
            config = DEFAULT_CONFIG
        
        if not config.enable_small_object_detection or len(detections) == 0:
            return detections
        
        # Separate small and large objects
        small_objs, large_objs = SmallObjectDetector.filter_by_size(
            detections,
            image_width,
            image_height,
            config,
        )
        
        if not small_objs:
            return detections
        
        # Apply small-object specific filters
        small_objs = SmallObjectDetector._filter_by_shape(small_objs, config)
        small_objs = SmallObjectDetector._filter_by_dimensions(small_objs, config)
        small_objs = SmallObjectDetector.boost_small_object_confidence(small_objs, image_width, image_height, boost_factor=1.10)
        
        # Combine with large objects
        return small_objs + large_objs
    
    @staticmethod
    def _filter_by_shape(
        detections: List[Detection],
        config: DetectionConfig,
    ) -> List[Detection]:
        """Filter out degenerate sliver bounding boxes while preserving aerial objects."""
        filtered = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            width = x2 - x1
            height = y2 - y1
            
            if width > 0 and height > 0:
                aspect = width / height
                # Only reject extreme sliver artifacts (e.g. aspect ratio < 0.04 or > 25.0)
                if 0.04 <= aspect <= 25.0:
                    filtered.append(detection)
        
        return filtered
    
    @staticmethod
    def _filter_by_dimensions(
        detections: List[Detection],
        config: DetectionConfig,
    ) -> List[Detection]:
        """Filter by minimum pixel dimension."""
        min_width = config.small_object_config.get("min_width_pixels", 2)
        min_height = config.small_object_config.get("min_height_pixels", 2)
        
        return [
            d for d in detections
            if (d.bbox[2] - d.bbox[0]) >= min_width and (d.bbox[3] - d.bbox[1]) >= min_height
        ]
    
    @staticmethod
    def boost_small_object_confidence(
        detections: List[Detection],
        image_width: int,
        image_height: int,
        boost_factor: float = 1.05,
    ) -> List[Detection]:
        """
        Slightly boost confidence of small objects (they're usually hard to detect).
        This helps maintain them through filtering without breaking calibration.
        """
        
        image_area = image_width * image_height
        small_threshold = image_area * 0.0035
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            area = (x2 - x1) * (y2 - y1)
            
            if area < small_threshold:
                # Gentle boost: cap at 0.99 to avoid breaking downstream logic
                detection.confidence = min(
                    detection.confidence * boost_factor,
                    0.99
                )
        
        return detections
    
    @staticmethod
    def get_adaptive_confidence_threshold(
        object_area_pixels: float,
        image_width: int,
        image_height: int,
        config: Optional[DetectionConfig] = None,
    ) -> float:
        """
        Get adaptive confidence threshold based on object size.
        Smaller objects get lower thresholds to improve recall.
        """
        
        if config is None:
            config = DEFAULT_CONFIG
        
        image_area = image_width * image_height
        area_ratio = object_area_pixels / image_area
        
        # Classify by size
        if area_ratio < 0.001:  # Tiny (< 0.1%)
            return config.small_object_config.get("confidence_threshold", 0.10)
        elif area_ratio < 0.0035:  # Small (0.1-0.35%)
            return 0.15
        elif area_ratio < 0.01:  # Medium-small (0.35-1%)
            return 0.20
        else:  # Larger objects
            return config.base_confidence_threshold
    
    @staticmethod
    def analyze_object_sizes(
        detections: List[Detection],
        image_width: int,
        image_height: int,
    ) -> dict:
        """Analyze size distribution of detected objects."""
        
        if not detections:
            return {
                "count": 0,
                "avg_area_ratio": 0,
                "tiny_count": 0,
                "small_count": 0,
                "medium_count": 0,
                "large_count": 0,
            }
        
        image_area = image_width * image_height
        areas = []
        
        tiny = 0
        small = 0
        medium = 0
        large = 0
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / image_area
            areas.append(area_ratio)
            
            if area_ratio < 0.001:
                tiny += 1
            elif area_ratio < 0.0035:
                small += 1
            elif area_ratio < 0.01:
                medium += 1
            else:
                large += 1
        
        import statistics
        
        return {
            "count": len(detections),
            "avg_area_ratio": statistics.mean(areas) if areas else 0,
            "median_area_ratio": statistics.median(areas) if areas else 0,
            "min_area_ratio": min(areas) if areas else 0,
            "max_area_ratio": max(areas) if areas else 0,
            "tiny_count": tiny,
            "small_count": small,
            "medium_count": medium,
            "large_count": large,
        }
    
    @staticmethod
    def recommend_config_for_objects(
        detections: List[Detection],
        image_width: int,
        image_height: int,
    ) -> DetectionConfig:
        """
        Recommend configuration based on detected object sizes.
        If many small objects, recommend accuracy-optimized config.
        """
        
        stats = SmallObjectDetector.analyze_object_sizes(
            detections,
            image_width,
            image_height,
        )
        
        # If >30% are small/tiny, use accuracy-optimized
        small_and_tiny = stats["small_count"] + stats["tiny_count"]
        small_fraction = small_and_tiny / max(stats["count"], 1)
        
        if small_fraction > 0.3:
            print(f"  Detected many small objects ({small_fraction:.1%}), using accuracy-optimized config")
            return DetectionConfig.accuracy_optimized()
        else:
            return DetectionConfig.balanced()
