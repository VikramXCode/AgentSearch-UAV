"""
Detection optimization examples and testing utilities.
Demonstrates the new optimization features and how to use them.
"""

from models.detector import DetectionEngine
from models.sahi_engine import SAHIEngine
from models.detection_config import DetectionConfig
from models.detection_quality_optimizer import DetectionQualityOptimizer
from models.small_object_detector import SmallObjectDetector
from PIL import Image


class DetectionOptimizationExamples:
    """Examples of using detection optimizations."""
    
    @staticmethod
    def example_1_default_detection():
        """Example 1: Default detection with automatic optimizations."""
        print("\n" + "="*60)
        print("EXAMPLE 1: Default Detection (Optimizations Applied)")
        print("="*60)
        
        # Just use detector normally - optimizations apply automatically
        detector = DetectionEngine()
        
        image_path = input("Enter image path: ").strip()
        target = input("Enter target object: ").strip()
        
        result = detector.detect(
            image_path=image_path,
            target=target,
        )
        
        print(f"\nResults:")
        print(f"  Raw detections: {len(result.raw_detections)}")
        print(f"  Filtered detections: {len(result.filtered_detections)}")
        print(f"  Inference time: {result.inference_time:.3f}s")
        
        # Show detection details
        for i, det in enumerate(result.filtered_detections[:5], 1):
            print(f"  {i}. {det.label} ({det.confidence:.2f})")
    
    @staticmethod
    def example_2_accuracy_optimized():
        """Example 2: Accuracy-optimized detection (find more objects)."""
        print("\n" + "="*60)
        print("EXAMPLE 2: Accuracy Optimized (Find More Objects)")
        print("="*60)
        
        config = DetectionConfig.accuracy_optimized()
        detector = DetectionEngine(config=config)
        
        image_path = input("Enter image path: ").strip()
        target = input("Enter target object: ").strip()
        
        result = detector.detect(
            image_path=image_path,
            target=target,
            config=config,
        )
        
        print(f"\nAccuracy-Optimized Results:")
        print(f"  Detections: {len(result.filtered_detections)}")
        print(f"  Inference time: {result.inference_time:.3f}s")
        
        # Analyze sizes
        stats = SmallObjectDetector.analyze_object_sizes(
            result.filtered_detections,
            result.image_width,
            result.image_height,
        )
        
        print(f"\nSize Distribution:")
        print(f"  Tiny (<0.1%): {stats['tiny_count']}")
        print(f"  Small (0.1-0.35%): {stats['small_count']}")
        print(f"  Medium (0.35-1%): {stats['medium_count']}")
        print(f"  Large (>1%): {stats['large_count']}")
        print(f"  Average area: {stats['avg_area_ratio']:.3%}")
    
    @staticmethod
    def example_3_speed_optimized():
        """Example 3: Speed-optimized detection (faster)."""
        print("\n" + "="*60)
        print("EXAMPLE 3: Speed Optimized (Faster Inference)")
        print("="*60)
        
        config = DetectionConfig.speed_optimized()
        detector = DetectionEngine(config=config)
        
        image_path = input("Enter image path: ").strip()
        target = input("Enter target object: ").strip()
        
        result = detector.detect(
            image_path=image_path,
            target=target,
            config=config,
        )
        
        print(f"\nSpeed-Optimized Results:")
        print(f"  Detections: {len(result.filtered_detections)}")
        print(f"  Inference time: {result.inference_time:.3f}s")
    
    @staticmethod
    def example_4_compare_configs():
        """Example 4: Compare different configurations side-by-side."""
        print("\n" + "="*60)
        print("EXAMPLE 4: Compare Configurations")
        print("="*60)
        
        image_path = input("Enter image path: ").strip()
        target = input("Enter target object: ").strip()
        
        configs = {
            "Speed Optimized": DetectionConfig.speed_optimized(),
            "Balanced": DetectionConfig.balanced(),
            "Accuracy Optimized": DetectionConfig.accuracy_optimized(),
        }
        
        detector = DetectionEngine()
        
        print(f"\nComparing configurations on: {image_path}")
        print(f"Target: {target}")
        print("\n" + "-"*60)
        
        results = {}
        for name, config in configs.items():
            result = detector.detect(
                image_path=image_path,
                target=target,
                config=config,
            )
            results[name] = result
            print(f"\n{name}:")
            print(f"  Detections: {len(result.filtered_detections)}")
            print(f"  Time: {result.inference_time:.3f}s")
        
        # Analysis
        print("\n" + "-"*60)
        print("Analysis:")
        for name, result in results.items():
            avg_conf = sum(d.confidence for d in result.filtered_detections) / max(len(result.filtered_detections), 1)
            print(f"  {name}: {len(result.filtered_detections)} objects, avg confidence {avg_conf:.2f}")
    
    @staticmethod
    def example_5_soft_nms_comparison():
        """Example 5: Compare soft-NMS vs hard NMS."""
        print("\n" + "="*60)
        print("EXAMPLE 5: Soft-NMS vs Hard NMS")
        print("="*60)
        
        image_path = input("Enter image path: ").strip()
        target = input("Enter target object: ").strip()
        
        # Hard NMS
        config_hard = DetectionConfig.balanced()
        config_hard.use_soft_nms = False
        
        # Soft-NMS Linear
        config_soft_linear = DetectionConfig.balanced()
        config_soft_linear.use_soft_nms = True
        config_soft_linear.soft_nms_method = "linear"
        
        # Soft-NMS Gaussian
        config_soft_gaussian = DetectionConfig.balanced()
        config_soft_gaussian.use_soft_nms = True
        config_soft_gaussian.soft_nms_method = "gaussian"
        
        detector = DetectionEngine()
        
        print("\nRunning detection with different NMS methods...")
        
        result_hard = detector.detect(image_path, target, config=config_hard)
        result_soft_linear = detector.detect(image_path, target, config=config_soft_linear)
        result_soft_gaussian = detector.detect(image_path, target, config=config_soft_gaussian)
        
        print("\n" + "-"*60)
        print("Results:")
        print(f"  Hard NMS: {len(result_hard.filtered_detections)} detections")
        print(f"  Soft-NMS (Linear): {len(result_soft_linear.filtered_detections)} detections")
        print(f"  Soft-NMS (Gaussian): {len(result_soft_gaussian.filtered_detections)} detections")
    
    @staticmethod
    def example_6_sahi_optimization():
        """Example 6: SAHI with optimized slicing."""
        print("\n" + "="*60)
        print("EXAMPLE 6: SAHI with Optimizations")
        print("="*60)
        
        image_path = input("Enter image path: ").strip()
        target = input("Enter target object: ").strip()
        
        # Accuracy-optimized SAHI
        config = DetectionConfig.accuracy_optimized()
        sahi = SAHIEngine()
        
        print(f"\nRunning SAHI with accuracy-optimized config...")
        print(f"  Small object mode: {config.sahi_small_object_mode}")
        print(f"  Overlap boost: {config.sahi_small_object_overlap_boost}x")
        
        detections = sahi.detect(image_path, target, config=config)
        
        print(f"\nResults:")
        print(f"  Detections: {len(detections)}")
        
        # Analyze
        from models.schemas import Detection
        dets = [
            Detection(
                label=d["class"],
                confidence=d["confidence"],
                bbox=d["bbox"]
            )
            for d in detections
        ]
        
        img = Image.open(image_path)
        stats = SmallObjectDetector.analyze_object_sizes(dets, img.width, img.height)
        print(f"\nSize distribution:")
        print(f"  Tiny: {stats['tiny_count']}, Small: {stats['small_count']}, "
              f"Medium: {stats['medium_count']}, Large: {stats['large_count']}")
    
    @staticmethod
    def example_7_custom_config():
        """Example 7: Create a custom configuration."""
        print("\n" + "="*60)
        print("EXAMPLE 7: Custom Configuration")
        print("="*60)
        
        # Start with balanced config
        config = DetectionConfig.balanced()
        
        # Customize for specific scenario
        print("\nCustomizing configuration for small UAV objects...")
        
        config.base_confidence_threshold = 0.15
        config.use_soft_nms = True
        config.soft_nms_method = "gaussian"
        config.sahi_small_object_mode = True
        config.sahi_small_object_overlap_boost = 1.8
        config.enable_small_object_detection = True
        config.min_area_ratio = 0.0005  # Very small objects OK
        
        print(f"\nCustom config settings:")
        print(f"  Confidence threshold: {config.base_confidence_threshold}")
        print(f"  Soft-NMS: {config.soft_nms_method}")
        print(f"  SAHI overlap boost: {config.sahi_small_object_overlap_boost}x")
        print(f"  Small object detection: {config.enable_small_object_detection}")
        
        image_path = input("\nEnter image path: ").strip()
        target = input("Enter target object: ").strip()
        
        detector = DetectionEngine(config=config)
        result = detector.detect(image_path, target, config=config)
        
        print(f"\nResults with custom config:")
        print(f"  Detections: {len(result.filtered_detections)}")
        print(f"  Inference time: {result.inference_time:.3f}s")
    
    @staticmethod
    def example_8_quality_analysis():
        """Example 8: Analyze and report detection quality."""
        print("\n" + "="*60)
        print("EXAMPLE 8: Quality Analysis")
        print("="*60)
        
        image_path = input("Enter image path: ").strip()
        target = input("Enter target object: ").strip()
        
        detector = DetectionEngine()
        result = detector.detect(image_path, target)
        
        # Optimize and get report
        optimized, report = DetectionQualityOptimizer.optimize_detections(
            result.filtered_detections,
            image_path,
            config=DEFAULT_CONFIG,
            verbose=True,
        )
        
        # Print report
        DetectionQualityOptimizer.print_optimization_report(report)
    
    @staticmethod
    def run_all_examples():
        """Run all examples interactively."""
        examples = {
            "1": ("Default Detection", DetectionOptimizationExamples.example_1_default_detection),
            "2": ("Accuracy Optimized", DetectionOptimizationExamples.example_2_accuracy_optimized),
            "3": ("Speed Optimized", DetectionOptimizationExamples.example_3_speed_optimized),
            "4": ("Compare Configs", DetectionOptimizationExamples.example_4_compare_configs),
            "5": ("Soft-NMS Comparison", DetectionOptimizationExamples.example_5_soft_nms_comparison),
            "6": ("SAHI Optimization", DetectionOptimizationExamples.example_6_sahi_optimization),
            "7": ("Custom Config", DetectionOptimizationExamples.example_7_custom_config),
            "8": ("Quality Analysis", DetectionOptimizationExamples.example_8_quality_analysis),
        }
        
        while True:
            print("\n" + "="*60)
            print("DETECTION OPTIMIZATION EXAMPLES")
            print("="*60)
            for key, (name, _) in examples.items():
                print(f"  {key}. {name}")
            print("  0. Exit")
            
            choice = input("\nSelect example (0-8): ").strip()
            
            if choice == "0":
                print("Exiting...")
                break
            
            if choice in examples:
                try:
                    examples[choice][1]()
                except Exception as e:
                    print(f"\nError: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Invalid selection")


if __name__ == "__main__":
    from models.detection_config import DEFAULT_CONFIG
    
    print("Detection Optimization Examples")
    print("================================\n")
    
    DetectionOptimizationExamples.run_all_examples()
