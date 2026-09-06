# Integration Guide for Enhanced Detectors

## Quick Start

### Option 1: Use Standard Detector (Baseline)
No changes needed - already using improved thresholds (0.60 confidence, 0.35 NMS).

```python
from models.detector import DetectionEngine

detector = DetectionEngine()
result = detector.detect(image_path="image.jpg", target="person")
```

---

### Option 2: Use Ensemble Detector (Best Accuracy)

Replace in your detection agent:

```python
from models.enhanced_detector import EnsembleDetector
from models.detection_verifier import DetectionVerifier

# Initialize
ensemble = EnsembleDetector()

# Detect
result = ensemble.detect(
    image_path=image_path,
    target=target,
    confidence=0.60,
    clip_threshold=0.25,
)

# Verify
result.filtered_detections = DetectionVerifier.apply_all_filters(
    detections=result.filtered_detections,
    image_width=result.image_width,
    image_height=result.image_height,
)
```

---

### Option 3: Use Super Resolution (Small Objects)

```python
from models.enhanced_detector import SuperResolutionDetector

sr_detector = SuperResolutionDetector(scale=2)

result = sr_detector.detect(
    image_path=image_path,
    target=target,
    use_sr=True,  # Enable super resolution
)
```

---

### Option 4: Use Adaptive Detector (Auto-Tuned)

```python
from models.enhanced_detector import AdaptiveDetector

adaptive = AdaptiveDetector()

result = adaptive.detect(
    image_path=image_path,
    target=target,
)
```

---

## Integration with Detection Agent

Update `agents/detection_agent.py`:

```python
from dataclasses import dataclass
from models.enhanced_detector import (
    EnsembleDetector,
    SuperResolutionDetector,
    AdaptiveDetector,
)
from models.detection_verifier import DetectionVerifier
from models.detector import DetectionEngine
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

    def __init__(
        self,
        model_path: str | None = None,
        mode: str = "adaptive",  # "standard", "ensemble", "superres", "adaptive"
    ):
        self.mode = mode
        self.model_path = model_path

        if mode == "standard":
            self.detector = DetectionEngine(model_path=model_path)
        elif mode == "ensemble":
            self.detector = EnsembleDetector(model_path=model_path)
        elif mode == "superres":
            self.detector = SuperResolutionDetector(model_path=model_path)
        elif mode == "adaptive":
            self.detector = AdaptiveDetector(model_path=model_path)
        else:
            raise ValueError(f"Unknown detection mode: {mode}")

    def run(self, state: AgentState, image_path: str) -> AgentState:

        print("\n==============================")
        print("    DETECTION AGENT")
        print(f"    Mode: {self.mode.upper()}")
        print("==============================")

        run_result = self._detect(image_path=image_path, state=state)

        state.detection.objects_found = [
            {
                "label": detection.label,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
            }
            for detection in run_result.detections
        ]

        state.detection.processed_image_path = run_result.processed_image_path
        state.detection.output_image_path = run_result.output_image_path
        state.strategy.detector = run_result.detector_name

        print(f"Detections: {len(state.detection.objects_found)}")

        return state

    def _detect(
        self,
        image_path: str,
        state: AgentState,
    ) -> DetectionRunResult:

        target = state.query.target

        # Run detection
        if self.mode == "ensemble":
            result = self.detector.detect(
                image_path,
                target,
                confidence=0.60,
                clip_threshold=0.25,
            )
        elif self.mode == "superres":
            result = self.detector.detect(
                image_path,
                target,
                use_sr=True,
                confidence=0.60,
            )
        else:
            result = self.detector.detect(image_path, target)

        # Apply verification filters
        verified = DetectionVerifier.apply_all_filters(
            detections=result.filtered_detections,
            image_width=result.image_width,
            image_height=result.image_height,
            min_confidence=0.60,
            check_edge_proximity=True,
        )

        # Visualize
        output_path = DETECTION_OUTPUT_PATH
        DetectionVisualizer.draw(
            image_path=image_path,
            detections=verified,
            output_path=output_path,
        )

        return DetectionRunResult(
            detections=verified,
            output_image_path=output_path,
            processed_image_path=image_path,
            detector_name=result.model_name,
        )
```

---

## Using in Main Workflow

Update `workflows/graph.py` or your main entry point:

```python
from workflows.state import AgentState
from agents.detection_agent import DetectionAgent

# Initialize with your chosen mode
detection_agent = DetectionAgent(mode="ensemble")  # or "superres", "adaptive"

# Use in pipeline
state = detection_agent.run(state, image_path="uav_image.jpg")
```

---

## Configuration File Approach

Create `config.yaml`:

```yaml
detection:
  mode: "ensemble"  # standard, ensemble, superres, adaptive
  confidence: 0.60
  
  ensemble:
    clip_threshold: 0.25
  
  superres:
    scale: 2
    enable: true
  
  verification:
    apply_confidence_filter: true
    apply_size_filter: true
    apply_aspect_ratio_filter: true
    check_edge_proximity: true
    min_confidence: 0.60
    min_area_ratio: 0.001
    max_area_ratio: 0.95
```

Load and use:

```python
import yaml
from agents.detection_agent import DetectionAgent

with open("config.yaml") as f:
    config = yaml.safe_load(f)

det_mode = config["detection"]["mode"]
detection_agent = DetectionAgent(mode=det_mode)
```

---

## Performance Tuning

### If getting too many false positives:
```python
# Increase confidence
detector = EnsembleDetector()
result = detector.detect(
    image_path,
    target,
    confidence=0.70,      # was 0.60
    clip_threshold=0.35,  # was 0.25
)

# Or use stricter verification
verified = DetectionVerifier.apply_all_filters(
    detections=result.filtered_detections,
    image_width=result.image_width,
    image_height=result.image_height,
    min_confidence=0.70,
    check_edge_proximity=True,
)
```

### If missing small objects:
```python
# Use super resolution
sr_detector = SuperResolutionDetector(scale=2)
result = sr_detector.detect(
    image_path,
    target,
    confidence=0.50,  # Lower confidence for recall
    use_sr=True,
)

# Or enable SAHI in detection agent
state.strategy.enable_sahi = True
```

### If inference is too slow:
```python
# Use standard detector without verification
detector = DetectionEngine()
result = detector.detect(image_path, target)

# Skip verification
state.skip_verification = True
```

---

## Benchmarking Your Setup

Test all detection modes on your dataset:

```python
import time
from models.enhanced_detector import (
    EnsembleDetector,
    SuperResolutionDetector,
    AdaptiveDetector,
)
from models.detector import DetectionEngine

test_image = "test_uav_image.jpg"
target = "person"

detectors = {
    "Standard": DetectionEngine(),
    "Ensemble": EnsembleDetector(),
    "SuperRes": SuperResolutionDetector(),
    "Adaptive": AdaptiveDetector(),
}

for name, detector in detectors.items():
    start = time.time()
    result = detector.detect(test_image, target)
    elapsed = time.time() - start
    
    print(f"{name:15} | "
          f"Detections: {len(result.filtered_detections):3} | "
          f"Time: {elapsed:.3f}s | "
          f"Model: {result.model_name}")
```

---

## Troubleshooting Integration

### ImportError: No module named 'enhanced_detector'
```bash
# Make sure you're running from project root
cd C:\path\to\AgentSearch-UAV
python -m your_script
```

### CLIP not installed
```bash
pip install clip-by-openai
# Or if that fails
pip install git+https://github.com/openai/CLIP.git
```

### SAHI not installed
```bash
pip install sahi
```

### Memory issues with ensemble detection
```python
# Reduce image size before detection
from PIL import Image

img = Image.open(image_path)
img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
img.save("downscaled.jpg")

# Then detect on downscaled image
result = ensemble.detect("downscaled.jpg", target)
```

---

## Backward Compatibility

All changes are backward compatible:
- Existing `DetectionEngine` still works (improved thresholds)
- `SAHIEngine` still works (improved thresholds + synonyms)
- New detectors are optional add-ons
- No changes required to existing code unless you want new features
