"""
Detection optimization configuration for SAHI, confidence thresholds, NMS, and small-object detection.
Provides tunable parameters for balancing accuracy vs. inference speed without breaking query-based architecture.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class DetectionConfig:
    """Main configuration for detection pipeline optimization."""
    
    # ===== Confidence Threshold Configuration =====
    # Base confidence thresholds for different detection modes
    base_confidence_threshold: float = 0.30  # General detection
    sahi_confidence_threshold: float = 0.55  # SAHI mode (slightly higher to avoid noise in slices)
    small_object_confidence_threshold: float = 0.30  # Small objects tolerance
    
    # Size-based confidence thresholds (object area as fraction of image)
    # Smaller objects get lower thresholds to avoid missing them
    confidence_by_size: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "tiny": (0.001, 0.10),      # < 0.1% of image
        "small": (0.10, 0.35),      # 0.1-0.35% of image
        "medium": (0.35, 1.0),      # 0.35-1% of image
        "large": (1.0, 50.0),       # 1-50% of image
    })
    
    # ===== NMS (Non-Maximum Suppression) Configuration =====
    # Standard NMS
    nms_iou_threshold: float = 0.35  # IoU threshold for merging boxes
    nms_score_threshold: float = 0.30  # Minimum confidence to keep detection
    
    # Soft-NMS configuration (better for overlapping/small objects)
    use_soft_nms: bool = True
    soft_nms_method: str = "linear"  # "linear", "gaussian", or "standard"
    soft_nms_iou_threshold: float = 0.35
    soft_nms_sigma: float = 0.5  # Gaussian parameter
    soft_nms_score_decay: float = 0.8  # Linear decay factor
    
    # Scale-aware NMS (different thresholds for different object sizes)
    use_scale_aware_nms: bool = True
    scale_aware_nms_config: Dict[str, float] = field(default_factory=lambda: {
        "tiny": 0.30,      # Tighter NMS for tiny objects (reduce duplicates)
        "small": 0.35,     # Standard for small objects
        "medium": 0.40,    # Slightly looser for medium objects
        "large": 0.50,     # Looser for large objects (less likely to be duplicates)
    })
    
    # ===== SAHI Configuration =====
    # Adaptive slice configuration for different image sizes
    sahi_enable: bool = True
    
    # Slice size configuration (height, width, overlap_h, overlap_w)
    sahi_slice_configs: Dict[str, Tuple[int, int, float, float]] = field(default_factory=lambda: {
        "ultra_large": (640, 640, 0.15, 0.15),   # 4M+ px: larger slices, small overlap for speed
        "large": (512, 512, 0.20, 0.20),          # 1.5M-4M px: balanced
        "medium": (384, 384, 0.25, 0.25),         # <1.5M px: smaller slices, more overlap
        "small_object_focused": (320, 320, 0.35, 0.35),  # High overlap for small objects
    })
    
    # Slice overlap increase for small-object detection
    sahi_small_object_mode: bool = True
    sahi_small_object_overlap_boost: float = 1.5  # Multiply overlap by this factor
    
    # ===== Small Object Detection Configuration =====
    enable_small_object_detection: bool = True
    
    # Minimum object size (as fraction of image area)
    small_object_min_area_ratio: float = 0.000005  # Allow tiny UAV objects down to ~10px
    
    # Aggressive small-object detection parameters
    small_object_config: Dict[str, float] = field(default_factory=lambda: {
        "confidence_threshold": 0.05,           # Very permissive for tiny objects
        "nms_iou_threshold": 0.30,             # Tighter to avoid duplicates
        "min_width_pixels": 2,                 # Minimum width in pixels
        "min_height_pixels": 2,                # Minimum height in pixels
        "max_width_multiplier": 0.95,          # Permissive for elongated objects
        "max_height_multiplier": 0.95,         # Permissive for elongated objects
    })
    
    # ===== Post-Detection Filtering =====
    # Area-based filtering
    use_size_filtering: bool = True
    min_area_ratio: float = 0.000005  # Minimum ~10px area for UAV benchmark objects
    max_area_ratio: float = 0.95     # Maximum 95% of image
    
    # Aspect ratio filtering (width/height)
    use_aspect_ratio_filtering: bool = False
    min_aspect_ratio: float = 0.05   # Very permissive for small/narrow objects
    max_aspect_ratio: float = 20.0   # Very permissive for wide objects
    
    # Spatial filtering (remove detections at image edges)
    use_edge_filtering: bool = False
    edge_margin_ratio: float = 0.005  # 0.5% margin from edges
    
    # ===== Detection Deduplication =====
    # Reduce duplicate detections from slice overlaps
    use_deduplication: bool = True
    dedup_iou_threshold: float = 0.70  # Merge duplicate detections from slice overlaps
    dedup_confidence_merge: bool = True  # Merge by averaging confidence
    
    # ===== Performance Settings =====
    # Inference optimization
    use_model_caching: bool = True  # Cache model between calls
    max_detections_per_image: int = 1000  # Maximum detections to keep
    
    # Multi-threading for SAHI
    sahi_num_threads: int = 1  # Set > 1 for parallel slice processing
    
    # ===== Vocabulary Building =====
    # Better vocabulary for YOLO-World
    use_extended_synonyms: bool = True
    expand_vocabulary: bool = True
    
    # Class-specific configurations
    class_specific_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "person": 0.25,           # People usually clear
        "vehicle": 0.20,          # Vehicles can be small/occluded
        "aircraft": 0.15,         # Aircraft harder to detect at distance
        "small_object": 0.10,     # Fallback for unrecognized small objects
    })
    
    # ===== Optimization Presets =====
    @staticmethod
    def accuracy_optimized() -> 'DetectionConfig':
        """Optimized for maximum accuracy at cost of speed."""
        config = DetectionConfig()
        config.base_confidence_threshold = 0.15
        config.small_object_confidence_threshold = 0.08
        config.use_soft_nms = True
        config.soft_nms_method = "gaussian"
        config.sahi_enable = True
        config.sahi_small_object_mode = True
        config.sahi_small_object_overlap_boost = 2.0
        config.enable_small_object_detection = True
        config.use_size_filtering = True
        config.use_aspect_ratio_filtering = True
        config.max_detections_per_image = 2000
        return config
    
    @staticmethod
    def balanced() -> 'DetectionConfig':
        """Balanced between accuracy and speed (default)."""
        config = DetectionConfig()
        config.base_confidence_threshold = 0.10
        config.small_object_confidence_threshold = 0.05
        config.use_soft_nms = True
        config.soft_nms_method = "linear"
        config.use_size_filtering = True
        config.min_area_ratio = 0.000005
        config.use_aspect_ratio_filtering = False
        config.use_edge_filtering = False
        return config

    @staticmethod
    def uav_benchmark() -> 'DetectionConfig':
        """Optimized for UAV aerial benchmarks (VisDrone small objects)."""
        config = DetectionConfig()
        config.base_confidence_threshold = 0.08
        config.nms_score_threshold = 0.08
        config.small_object_confidence_threshold = 0.04
        config.use_soft_nms = True
        config.soft_nms_method = "linear"
        config.soft_nms_iou_threshold = 0.35
        config.soft_nms_score_decay = 0.50
        config.use_size_filtering = True
        config.min_area_ratio = 0.000005
        config.max_area_ratio = 0.95
        config.use_aspect_ratio_filtering = False
        config.use_edge_filtering = False
        config.use_deduplication = True
        config.dedup_iou_threshold = 0.65
        config.sahi_slice_configs = {
            "ultra_large": (960, 960, 0.20, 0.20),
            "large": (960, 960, 0.20, 0.20),
            "medium": (960, 960, 0.20, 0.20),
            "small_object_focused": (960, 960, 0.20, 0.20),
        }
        return config
    
    @staticmethod
    def speed_optimized() -> 'DetectionConfig':
        """Optimized for speed, reasonable accuracy."""
        config = DetectionConfig()
        config.base_confidence_threshold = 0.30
        config.small_object_confidence_threshold = 0.20
        config.use_soft_nms = False  # Disable soft-NMS for speed
        config.sahi_enable = True
        config.sahi_small_object_mode = False
        config.enable_small_object_detection = False
        config.use_size_filtering = True
        config.use_aspect_ratio_filtering = False
        config.max_detections_per_image = 500
        return config
    
    def get_slice_config(self, image_width: int, image_height: int, small_object_focus: bool = False) -> Tuple[int, int, float, float]:
        """Get optimal SAHI slice configuration for image size."""
        if not self.sahi_enable:
            return image_width, image_height, 0.0, 0.0
        
        area = image_width * image_height
        max_side = max(image_width, image_height)
        
        # Select base config
        if area >= 4_000_000 or max_side >= 2400:
            config_name = "ultra_large"
        elif area >= 1_500_000 or max_side >= 1600:
            config_name = "large"
        elif small_object_focus and self.sahi_small_object_mode:
            config_name = "small_object_focused"
        else:
            config_name = "medium"
        
        slice_h, slice_w, overlap_h, overlap_w = self.sahi_slice_configs[config_name]
        
        # Apply boost for small objects
        if small_object_focus and self.sahi_small_object_mode:
            boost = self.sahi_small_object_overlap_boost
            overlap_h = min(0.5, overlap_h * boost)  # Cap at 50% to maintain efficiency
            overlap_w = min(0.5, overlap_w * boost)
        
        return slice_h, slice_w, overlap_h, overlap_w
    
    def get_confidence_threshold(self, object_area_ratio: float) -> float:
        """Get confidence threshold based on object size."""
        return self.base_confidence_threshold
    
    def get_nms_threshold(self, object_area_ratio: float) -> float:
        """Get NMS IoU threshold based on object size."""
        if not self.use_scale_aware_nms:
            return self.nms_iou_threshold
        
        # Determine size category
        if object_area_ratio < 0.10:
            size_category = "tiny"
        elif object_area_ratio < 0.35:
            size_category = "small"
        elif object_area_ratio < 1.0:
            size_category = "medium"
        else:
            size_category = "large"
        
        return self.scale_aware_nms_config.get(size_category, self.nms_iou_threshold)


# Global default configuration
DEFAULT_CONFIG = DetectionConfig()
