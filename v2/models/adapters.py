from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from v2.schemas.state import QuerySpec, Candidate, MediaMetadata
import uuid

class BaseDetectorAdapter(ABC):
    @abstractmethod
    def detect(self, image: Any, query_spec: QuerySpec, metadata: MediaMetadata) -> List[Candidate]:
        """
        Runs detection and returns normalized candidates.
        """
        pass
        
    def _normalize_candidate(self, bbox: List[float], conf: float, label: str, source: str, frame_id: Optional[int] = None) -> Candidate:
        """
        Canonical normalization layer. 
        """
        # Ensure confidence is clamped
        conf = max(0.0, min(1.0, conf))
        
        # Ensure valid bbox
        if len(bbox) != 4:
            raise ValueError(f"Invalid bbox length: {len(bbox)}")
        
        x1, y1, x2, y2 = bbox
        if x1 > x2 or y1 > y2:
            raise ValueError(f"Invalid bbox coordinates: {bbox}")
            
        return Candidate(
            id=str(uuid.uuid4()),
            bbox=[x1, y1, x2, y2],
            class_label=label,
            confidence=conf,
            source=source,
            frame_id=frame_id
        )

class SpecialistDetectorAdapter(BaseDetectorAdapter):
    """
    Adapter for P2 (EXP11) closed-vocabulary aerial detector.
    """
    def __init__(self, model_instance: Any = None):
        self.model = model_instance
        self.source_name = "SPECIALIST_P2"
        
    def detect(self, image: Any, query_spec: QuerySpec, metadata: MediaMetadata) -> List[Candidate]:
        if not self.model:
            raise RuntimeError("Model instance not loaded.")
        
        # In a real implementation, this would call self.model.predict(...)
        # and parse the ultralytics results. 
        # Since we use mocks in unit tests, we'll delegate to a method we can mock.
        raw_results = self._run_inference(image, query_spec)
        
        candidates = []
        # frame_id logic if processing a video frame
        # If metadata has total_frames but we process 1, maybe it's passed via query_spec or we just leave None
        frame_id = metadata.fps if metadata.fps else None # simplified
        
        # Map target to valid model classes
        valid_classes = set()
        if query_spec.target:
            t = query_spec.target.lower()
            if t == "person" or t == "pedestrian":
                valid_classes.update(["pedestrian", "people", "person"])
            elif t == "motorcycle" or t == "bike" or t == "motor":
                valid_classes.update(["motor", "motorcycle"])
            elif t == "bicycle":
                valid_classes.update(["bicycle"])
            elif t == "car":
                valid_classes.update(["car"])
            elif t == "bus":
                valid_classes.update(["bus"])
            elif t == "truck":
                valid_classes.update(["truck"])
            elif t == "van":
                valid_classes.update(["van"])
            elif "tricycle" in t:
                valid_classes.update(["tricycle", "awning-tricycle"])
            else:
                valid_classes.add(t)
        
        for res in raw_results:
            if valid_classes and res["label"].lower() not in valid_classes and query_spec.target != "object":
                continue
                
            try:
                candidates.append(self._normalize_candidate(
                    bbox=res["bbox"],
                    conf=res["conf"],
                    label=res["label"],
                    source=self.source_name,
                    frame_id=None # We will mock frame_id logic inside tracking agent
                ))
            except ValueError as e:
                # Log rejection, but continue
                continue
                
        return candidates
        
    def _run_inference(self, image: Any, query_spec: QuerySpec) -> List[Dict]:
        """Mockable hook for ultralytics inference. If real model, call it."""
        if isinstance(self.model, str):
            return [] # Mock
            
        results = self.model(image, verbose=False)
        out = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                label = self.model.names[int(cls)]
                out.append({
                    "bbox": box.tolist(),
                    "conf": float(conf),
                    "label": label
                })
        return out

class OpenWorldDetectorAdapter(BaseDetectorAdapter):
    """
    Adapter for YOLO-World open-vocabulary detector.
    """
    def __init__(self, model_instance: Any = None):
        self.model = model_instance
        self.source_name = "OPEN_WORLD"
        
    def detect(self, image: Any, query_spec: QuerySpec, metadata: MediaMetadata) -> List[Candidate]:
        if not self.model:
            raise RuntimeError("Model instance not loaded.")
            
        # Extract vocabulary from QuerySpec
        vocab = []
        if query_spec.target and query_spec.target != "object":
            vocab.append(query_spec.target)
        elif query_spec.raw_query:
            vocab.append(query_spec.raw_query)
            
        for constraint in query_spec.constraints:
            if constraint.constraint_type == "attribute":
                vocab.append(constraint.value)
            
        if not vocab:
            vocab = ["object"]
            
        # In a real implementation, we would set YOLO-World classes to vocab here.
        raw_results = self._run_inference(image, vocab)
        
        candidates = []
        for res in raw_results:
            try:
                candidates.append(self._normalize_candidate(
                    bbox=res["bbox"],
                    conf=res["conf"],
                    label=res["label"],
                    source=self.source_name
                ))
            except ValueError:
                continue
                
        return candidates
        
    def _run_inference(self, image: Any, vocab: List[str]) -> List[Dict]:
        """Mockable hook for ultralytics inference. If real model, call it."""
        if isinstance(self.model, str):
            return [] # Mock
            
        self.model.set_classes(vocab)
        results = self.model(image, verbose=False, conf=0.05)
        out = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                label = self.model.names[int(cls)]
                out.append({
                    "bbox": box.tolist(),
                    "conf": float(conf),
                    "label": label
                })
        return out
