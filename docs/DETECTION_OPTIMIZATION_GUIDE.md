# Detection Pipeline Optimization Guide

## Overview

This guide explains the comprehensive detection pipeline optimizations implemented to improve object detection accuracy without sacrificing inference speed or breaking the existing query-based architecture.

## Key Optimizations

### 1. **Soft-NMS (Non-Maximum Suppression)**
**What**: Replaced hard NMS with configurable soft-NMS that uses confidence decay instead of hard removal.

**Why**: Soft-NMS is better for:
- Overlapping objects (especially in SAHI slices)
- Small objects that may have partial overlaps
- Crowded scenes with dense object distributions

**Methods**:
- **Linear**: `confidence *= (1 - IoU * decay_factor)` - Good balance
- **Gaussian**: `confidence *= exp(-IoU² / σ²)` - Smoother decay, better for close objects
- **Standard**: Hard NMS with threshold - Fast but may lose valid detections

**Configuration**:
```python
config = DetectionConfig()
config.use_soft_nms = True
config.soft_nms_method = "linear"  # or "gaussian"
config.soft_nms_iou_threshold = 0.35
config.soft_nms_score_decay = 0.8  # Linear only
config.soft_nms_sigma = 0.5        # Gaussian only
```

### 2. **Scale-Aware NMS**
**What**: Different NMS thresholds for objects of different sizes.

**Why**: 
- Tiny objects (< 0.1%): IOU 0.30 - Tighter to reduce duplicates from slicing
- Small objects (0.1-0.35%): IOU 0.35 - Standard threshold
- Medium objects (0.35-1%): IOU 0.40 - Slightly looser
- Large objects (> 1%): IOU 0.50 - Looser (less likely to be real duplicates)

**Enable**:
```python
config.use_scale_aware_nms = True
config.scale_aware_nms_config = {
    "tiny": 0.30,
    "small": 0.35,
    "medium": 0.40,
    "large": 0.50,
}
```

### 3. **Size-Based Confidence Thresholds**
**What**: Automatically lower confidence thresholds for smaller objects.

**Thresholds by size**:
- Tiny (< 0.1%): 0.10 - Very permissive
- Small (0.1-0.35%): 0.15 - Permissive
- Medium (0.35-1%): 0.20 - Standard
- Large (> 1%): 0.30 - Strict (more likely to be true positive)

**Why**: Small objects are harder to detect accurately, so lower thresholds help find them without excessive false positives.

**Use**:
```python
config.confidence_by_size = {
    "tiny": (0.001, 0.10),      # min%, max%, threshold
    "small": (0.10, 0.35),
    "medium": (0.35, 1.0),
    "large": (1.0, 50.0),
}
```

### 4. **Optimized SAHI Configuration**
**What**: Adaptive slice configuration based on image size and detection focus.

**Preset configurations**:

| Scene | Slice Size | Overlap | Purpose |
|-------|-----------|---------|---------|
| Ultra-large (4M+ px) | 640×640 | 15% | Speed optimized |
| Large (1.5-4M px) | 512×512 | 20% | Balanced |
| Medium (<1.5M px) | 384×384 | 25% | Detail focused |
| Small object mode | 320×320 | 35-52.5% | Small object coverage |

**Enable small-object focused mode**:
```python
config.sahi_small_object_mode = True
config.sahi_small_object_overlap_boost = 1.5  # Multiply overlap by 1.5x
```

**How it works**:
- Smaller slices with more overlap = better coverage of small objects
- Larger slices with minimal overlap = faster inference
- System automatically detects if many small objects are present and adjusts

### 5. **Multi-Method Filtering Pipeline**
**What**: Series of specialized filters applied in sequence to maximize accuracy.

**Pipeline (in order)**:
1. **Aspect Ratio Filtering** - Remove extremely elongated detections
2. **Size Filtering** - Remove too-small or too-large objects
3. **Edge Filtering** - Remove partial detections at image boundaries
4. **Small Object Enhancement** - Apply shape analysis to small objects
5. **Soft-NMS** - Merge overlapping detections
6. **Deduplication** - Remove SAHI slice duplicates (IOU > 0.85)

**Enable/disable**:
```python
config.use_aspect_ratio_filtering = True
config.use_size_filtering = True
config.use_edge_filtering = True
config.enable_small_object_detection = True
config.use_soft_nms = True
config.use_deduplication = True
```

### 6. **Small Object Detection Optimization**
**What**: Specialized handling for tiny/distant objects.

**Features**:
- Shape validation (aspect ratio within reasonable bounds)
- Minimum dimension requirements
- Confidence boosting for hard-to-detect objects
- Adaptive threshold selection

**Configuration**:
```python
config.enable_small_object_detection = True
config.small_object_min_area_ratio = 0.001  # 0.1% of image

config.small_object_config = {
    "confidence_threshold": 0.10,
    "nms_iou_threshold": 0.25,        # Tighter for small objects
    "min_width_pixels": 4,            # Minimum 4px
    "min_height_pixels": 4,           # Minimum 4px
    "max_width_multiplier": 0.8,      # Don't detect if too wide
    "max_height_multiplier": 0.8,     # Don't detect if too tall
}
```

### 7. **Detection Deduplication**
**What**: Removes duplicate detections from SAHI slice overlaps intelligently.

**How it works**:
- Groups detections with very high IoU (0.85+)
- Optionally merges confidence scores
- Preserves high-confidence detections

**Configuration**:
```python
config.use_deduplication = True
config.dedup_iou_threshold = 0.85    # Very high = same object
config.dedup_confidence_merge = True  # Average confidences
```

### 8. **Quality Optimization Presets**

#### Accuracy Optimized
Best for finding all objects, regardless of speed.
```python
config = DetectionConfig.accuracy_optimized()
```

**Settings**:
- Lower confidence thresholds (0.15, 0.08)
- Soft-NMS with Gaussian decay
- SAHI enabled with 2x overlap boost
- Small object detection enabled
- Max 2000 detections

**Best for**: UAV imagery, crowded scenes, small objects

#### Balanced (Default)
Good balance between accuracy and speed.
```python
config = DetectionConfig.balanced()
```

**Settings**:
- Moderate confidence thresholds (0.20, 0.15)
- Soft-NMS with linear decay
- SAHI with standard overlap
- Small object detection enabled
- Max 1000 detections

**Best for**: General purpose, most UAV scenarios

#### Speed Optimized
Fastest inference, reasonable accuracy.
```python
config = DetectionConfig.speed_optimized()
```

**Settings**:
- Higher confidence thresholds (0.30, 0.20)
- No soft-NMS (standard NMS only)
- SAHI disabled for small objects
- Small object detection disabled
- Max 500 detections

**Best for**: Real-time processing, simple scenes

## Usage Examples

### Example 1: Using with Standard Detection Engine

```python
from models.detector import DetectionEngine
from models.detection_config import DetectionConfig

# Create detector with accuracy-optimized config
config = DetectionConfig.accuracy_optimized()
detector = DetectionEngine(config=config)

# Run detection
result = detector.detect(
    image_path="path/to/image.jpg",
    target="person",
    config=config  # Optional: can override at call time
)

print(f"Found {len(result.filtered_detections)} objects")
```

### Example 2: Using with SAHI

```python
from models.sahi_engine import SAHIEngine
from models.detection_config import DetectionConfig

config = DetectionConfig.accuracy_optimized()
sahi = SAHIEngine()

detections = sahi.detect(
    image_path="path/to/image.jpg",
    target="vehicle",
    config=config
)
```

### Example 3: Custom Configuration

```python
config = DetectionConfig()

# Customize for your specific use case
config.base_confidence_threshold = 0.18
config.use_soft_nms = True
config.soft_nms_method = "gaussian"
config.sahi_small_object_mode = True
config.sahi_small_object_overlap_boost = 1.8
config.min_aspect_ratio = 0.15
config.max_aspect_ratio = 8.0

detector = DetectionEngine(config=config)
result = detector.detect("image.jpg", "aircraft", config=config)
```

### Example 4: Automatic Configuration Selection

```python
from models.detection_quality_optimizer import DetectionQualityOptimizer

# Get recommendations based on image
config, recommendations = DetectionQualityOptimizer.analyze_and_recommend_config(
    image_path="path/to/image.jpg"
)

print(f"Recommended config: {config}")
print(f"Factors: {recommendations['factors']}")
```

### Example 5: Analysis and Optimization

```python
from models.detection_quality_optimizer import DetectionQualityOptimizer
from models.detection_config import DetectionConfig

# Run detection
result = detector.detect("image.jpg", "person")

# Optimize and analyze
optimized, report = DetectionQualityOptimizer.optimize_detections(
    result.filtered_detections,
    "image.jpg",
    config=DetectionConfig.balanced(),
    verbose=True
)

# Print report
DetectionQualityOptimizer.print_optimization_report(report)
```

## Architecture Integration

### Query-Based Architecture Compatibility
✅ **All optimizations preserve the query-based architecture**:
- Configuration is passed through the state system
- Detection Agent passes config to engines
- Engines apply optimizations transparently
- No changes to workflow or agent coordination

### State Integration
```python
# In StrategyAgent or workflow
state.strategy.confidence_threshold = 0.20
state.strategy.enable_sahi = True
state.strategy.enable_small_object_detection = True

# DetectionAgent uses these settings
# No need to change agent logic
```

## Performance Considerations

### Inference Speed Impact

| Feature | Speed Impact | Recommendation |
|---------|-------------|-----------------|
| Soft-NMS | ~5-10% slower | Worth it for accuracy |
| Scale-aware NMS | Negligible | Always enable |
| Size filtering | Negligible | Always enable |
| SAHI small object mode | ~20% slower | Enable when needed |
| Deduplication | ~2-5% slower | Always enable |

### Memory Impact
- Soft-NMS: Minimal additional memory
- Configuration objects: Negligible
- Detection storage: Same as before

### Recommended for Speed
If speed is critical:
```python
config = DetectionConfig.speed_optimized()
```

## Tuning Guide

### For Detecting Small Objects
```python
config = DetectionConfig.accuracy_optimized()
config.sahi_small_object_overlap_boost = 2.0  # More overlap
config.small_object_config["confidence_threshold"] = 0.08
config.min_area_ratio = 0.0001  # Allow very small objects
```

### For Dense Scenes (Many Objects)
```python
config = DetectionConfig.balanced()
config.use_soft_nms = True
config.soft_nms_method = "gaussian"  # Better for overlapping
config.soft_nms_sigma = 0.3  # Tighter Gaussian
```

### For High-Speed Real-Time Processing
```python
config = DetectionConfig.speed_optimized()
config.base_confidence_threshold = 0.35
config.max_detections_per_image = 200  # Limit output
```

### For UAV Imagery
```python
config = DetectionConfig.balanced()
config.sahi_enable = True
config.sahi_small_object_mode = True
config.enable_small_object_detection = True
config.edge_filtering = False  # UAV images often have objects at edges
config.max_detections_per_image = 1500
```

## Monitoring and Analysis

### Check Detection Quality
```python
from models.small_object_detector import SmallObjectDetector

stats = SmallObjectDetector.analyze_object_sizes(
    detections,
    image_width,
    image_height
)

print(f"Total: {stats['count']}")
print(f"Tiny: {stats['tiny_count']}, Small: {stats['small_count']}")
print(f"Avg area: {stats['avg_area_ratio']:.4%}")
```

### Compare Before/After
```python
DetectionQualityOptimizer.compare_before_after(
    before_detections,
    after_detections,
    image_width,
    image_height
)
```

## Common Issues and Solutions

### Issue: Many False Positives
**Solution**:
```python
config.base_confidence_threshold = 0.25  # Increase threshold
config.soft_nms_score_decay = 0.9       # Stricter decay
config.use_edge_filtering = True
```

### Issue: Missing Small Objects
**Solution**:
```python
config = DetectionConfig.accuracy_optimized()
config.sahi_small_object_overlap_boost = 2.0
config.small_object_config["confidence_threshold"] = 0.08
```

### Issue: Slow Inference
**Solution**:
```python
config = DetectionConfig.speed_optimized()
config.sahi_enable = False
config.max_detections_per_image = 300
```

### Issue: Duplicate Detections from SAHI
**Solution**:
```python
config.use_deduplication = True
config.dedup_iou_threshold = 0.80  # More aggressive
```

## Files and Modules

### Core Configuration
- `models/detection_config.py` - Main configuration class

### Optimizers
- `models/enhanced_postprocessor.py` - Soft-NMS and filtering
- `models/small_object_detector.py` - Small object optimization
- `models/detection_quality_optimizer.py` - Quality coordination

### Integration
- `models/detector.py` - Updated to use new optimizations
- `models/sahi_engine.py` - Updated to support configurations
- `agents/detection_agent.py` - Updated to pass configurations

## Next Steps

1. **Test with your data**: Run on actual UAV images
2. **Profile performance**: Measure inference times
3. **Tune thresholds**: Adjust based on your specific objects
4. **Monitor accuracy**: Track detection rates over time
5. **Use presets**: Start with accuracy/balanced/speed presets

## Questions?

Refer to test scripts in `tests/` directory for more examples.
