from pydantic import BaseModel
from pydantic.config import ConfigDict
from typing import List


class Detection(BaseModel):
    label: str
    confidence: float
    box: List[float]


class DetectionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_used: str
    inference_time_ms: float
    image_width: int
    image_height: int
    detections: List[Detection]


class HealthResponse(BaseModel):
    status: str
    models_available: List[str]
    device: str
