from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: Optional[int] = None
    source: str = "full_image"  # "full_image", "sahi", "enhanced_roi", "fused"
    scale_category: Optional[str] = None  # "tiny", "small", "medium", "large"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DetectionResult(BaseModel):
    model_name: str
    inference_time: float
    image_width: int
    image_height: int
    raw_detections: List[Detection] = Field(default_factory=list)
    filtered_detections: List[Detection] = Field(default_factory=list)
    latency_breakdown: Dict[str, float] = Field(default_factory=dict)
    pipeline_metadata: Dict[str, Any] = Field(default_factory=dict)