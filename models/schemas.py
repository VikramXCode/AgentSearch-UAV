from pydantic import BaseModel, Field
from typing import List


class Detection(BaseModel):

    label: str

    confidence: float

    bbox: List[float]


class DetectionResult(BaseModel):

    model_name: str

    inference_time: float

    image_width: int

    image_height: int

    raw_detections: List[Detection] = Field(default_factory=list)

    filtered_detections: List[Detection] = Field(default_factory=list)