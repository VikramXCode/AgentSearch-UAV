"""
Detection quality optimizer that coordinates all optimization strategies.
Intelligently selects between different detection modes and thresholds based on analysis.
"""

from typing import List, Optional, Tuple
from PIL import Image

from models.schemas import Detection
from models.detection_config import DetectionConfig, DEFAULT_CONFIG
from models.enhanced_postprocessor import EnhancedPostProcessor
from models.small_object_detector import SmallObjectDetector


class DetectionQualityOptimizer:
    """Orchestrates all detection quality improvements."""
    
    @staticmethod
    def optimize_detections(
        detections: List[Detection],
        image_path: str,
        config: Optional[DetectionConfig] = None,
        verbose: bool = True,
    ) -> Tuple[List[Detection], dict]:
        """
        Apply all optimizations in sequence.
        
        Returns:
            (optimized_detections, optimization_report)
        """
        
        if config is None:
            config = DEFAULT_CONFIG
        
        # Load image info
        img = Image.open(image_path)
        image_width, image_height = img.size
        
        report = {
            "input_count": len(detections),
            "optimizations_applied": [],
            "output_count": 0,
            "size_analysis": {},
        }
        
        if len(detections) == 0:
            return detections, report
        
        # Analyze input
        report["size_analysis"] = SmallObjectDetector.analyze_object_sizes(
            detections,
            image_width,
            image_height,
        )
        
        if verbose:
            print(f"\n  Input detections: {len(detections)}")
            print(f"  Size distribution: {report['size_analysis']['tiny_count']} tiny, "
                  f"{report['size_analysis']['small_count']} small, "
                  f"{report['size_analysis']['medium_count']} medium, "
                  f"{report['size_analysis']['large_count']} large")
        
        # 1. Apply aspect ratio filtering
        if config.use_aspect_ratio_filtering:
            before = len(detections)
            detections = EnhancedPostProcessor.apply_aspect_ratio_filtering(
                detections,
                config,
            )
            if len(detections) < before:
                report["optimizations_applied"].append(
                    f"Aspect ratio filtering: {before} -> {len(detections)}"
                )
        
        # 2. Apply size filtering
        if config.use_size_filtering:
            before = len(detections)
            detections = EnhancedPostProcessor.apply_size_filtering(
                detections,
                config,
                image_width,
                image_height,
            )
            if len(detections) < before:
                report["optimizations_applied"].append(
                    f"Size filtering: {before} -> {len(detections)}"
                )
        
        # 3. Apply edge filtering
        if config.use_edge_filtering:
            before = len(detections)
            detections = EnhancedPostProcessor.apply_edge_filtering(
                detections,
                config,
                image_width,
                image_height,
            )
            if len(detections) < before:
                report["optimizations_applied"].append(
                    f"Edge filtering: {before} -> {len(detections)}"
                )
        
        # 4. Enhance small object detection
        if config.enable_small_object_detection:
            before = len(detections)
            detections = SmallObjectDetector.enhance_small_object_detection(
                detections,
                image_width,
                image_height,
                config,
            )
            if len(detections) != before:
                report["optimizations_applied"].append(
                    f"Small object enhancement: {before} -> {len(detections)}"
                )
        
        # 5. Apply NMS (standard or soft)
        before = len(detections)
        detections = EnhancedPostProcessor.apply_nms(
            detections,
            config,
            image_width,
            image_height,
        )
        if len(detections) < before:
            nms_method = "Soft-NMS" if config.use_soft_nms else "Standard NMS"
            report["optimizations_applied"].append(
                f"{nms_method}: {before} -> {len(detections)}"
            )
        
        # 6. Limit detections if needed
        if config.max_detections_per_image and len(detections) > config.max_detections_per_image:
            before = len(detections)
            detections = detections[:config.max_detections_per_image]
            report["optimizations_applied"].append(
                f"Detection limit: {before} -> {len(detections)}"
            )
        
        report["output_count"] = len(detections)
        
        if verbose:
            print(f"\n  Optimizations:")
            for opt in report["optimizations_applied"]:
                print(f"    - {opt}")
            print(f"  Final detections: {len(detections)}")
        
        return detections, report
    
    @staticmethod
    def analyze_and_recommend_config(
        image_path: str,
        initial_detections: Optional[List[Detection]] = None,
    ) -> Tuple[DetectionConfig, dict]:
        """
        Analyze image and detections, recommend optimal configuration.
        """
        
        img = Image.open(image_path)
        image_width, image_height = img.size
        image_area = image_width * image_height
        
        recommendations = {
            "image_size": f"{image_width}x{image_height}",
            "image_area": image_area,
            "factors": [],
        }
        
        # Factor 1: Image size
        if image_area >= 4_000_000:
            recommendations["factors"].append("Very large image (4M+ pixels)")
            base_config = DetectionConfig.balanced()
        else:
            recommendations["factors"].append("Standard image size")
            base_config = DetectionConfig.balanced()
        
        # Factor 2: Object size distribution
        if initial_detections:
            stats = SmallObjectDetector.analyze_object_sizes(
                initial_detections,
                image_width,
                image_height,
            )
            
            small_fraction = (stats["small_count"] + stats["tiny_count"]) / max(stats["count"], 1)
            
            if small_fraction > 0.3:
                recommendations["factors"].append(f"Many small objects ({small_fraction:.1%})")
                base_config = DetectionConfig.accuracy_optimized()
            elif stats["count"] > 100:
                recommendations["factors"].append("Dense object distribution")
                base_config = DetectionConfig.balanced()
        
        return base_config, recommendations
    
    @staticmethod
    def get_confidence_threshold_for_size(
        bbox: List[float],
        image_width: int,
        image_height: int,
        config: Optional[DetectionConfig] = None,
    ) -> float:
        """Get adaptive confidence threshold for specific detection."""
        
        if config is None:
            config = DEFAULT_CONFIG
        
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        image_area = image_width * image_height
        area_ratio = area / image_area
        
        return config.get_confidence_threshold(area_ratio)
    
    @staticmethod
    def print_optimization_report(report: dict) -> None:
        """Print human-readable optimization report."""
        
        print("\n" + "="*50)
        print("DETECTION OPTIMIZATION REPORT")
        print("="*50)
        
        print(f"\nInput detections: {report['input_count']}")
        print(f"Output detections: {report['output_count']}")
        
        if report["size_analysis"]:
            stats = report["size_analysis"]
            print(f"\nSize distribution:")
            print(f"  Tiny (<0.1%):   {stats.get('tiny_count', 0):3d}")
            print(f"  Small (0.1-0.35%): {stats.get('small_count', 0):3d}")
            print(f"  Medium (0.35-1%): {stats.get('medium_count', 0):3d}")
            print(f"  Large (>1%):    {stats.get('large_count', 0):3d}")
            print(f"  Average area ratio: {stats.get('avg_area_ratio', 0):.4%}")
        
        if report["optimizations_applied"]:
            print(f"\nOptimizations applied:")
            for opt in report["optimizations_applied"]:
                print(f"  • {opt}")
        
        print()
    
    @staticmethod
    def compare_before_after(
        before_detections: List[Detection],
        after_detections: List[Detection],
        image_width: int,
        image_height: int,
    ) -> None:
        """Compare detection quality before and after optimization."""
        
        print("\n" + "="*50)
        print("OPTIMIZATION COMPARISON")
        print("="*50)
        
        before_stats = SmallObjectDetector.analyze_object_sizes(
            before_detections,
            image_width,
            image_height,
        )
        
        after_stats = SmallObjectDetector.analyze_object_sizes(
            after_detections,
            image_width,
            image_height,
        )
        
        print(f"\nBefore: {before_stats['count']} detections")
        print(f"  Tiny: {before_stats['tiny_count']}, Small: {before_stats['small_count']}, "
              f"Medium: {before_stats['medium_count']}, Large: {before_stats['large_count']}")
        print(f"  Avg area ratio: {before_stats['avg_area_ratio']:.4%}")
        
        print(f"\nAfter: {after_stats['count']} detections")
        print(f"  Tiny: {after_stats['tiny_count']}, Small: {after_stats['small_count']}, "
              f"Medium: {after_stats['medium_count']}, Large: {after_stats['large_count']}")
        print(f"  Avg area ratio: {after_stats['avg_area_ratio']:.4%}")
        
        change = after_stats['count'] - before_stats['count']
        pct_change = (change / max(before_stats['count'], 1)) * 100
        print(f"\nChange: {change:+d} ({pct_change:+.1f}%)")
        print()
