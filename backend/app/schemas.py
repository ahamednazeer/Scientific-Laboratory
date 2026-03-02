from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InstrumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    category: str
    description: str
    operation_steps: str
    safety_warnings: str


class DetectionRequest(BaseModel):
    image_b64: str = Field(..., description="Raw base64 JPEG/PNG string. Optional data URI prefix is supported.")
    run_refinement: bool = True
    mode: Literal["model", "ai"] = "model"


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    label: str
    confidence: float
    bbox: DetectionBox
    smoothed_bbox: DetectionBox
    stability_score: float
    instrument_id: int | None = None
    description: str | None = None
    operation_steps: str | None = None
    safety_warnings: str | None = None


class DetectionResponse(BaseModel):
    timestamp: datetime
    latency_ms: float
    fps_estimate: float
    detection_source_used: str
    detection_source_note: str | None = None
    detections: list[DetectionOut]


class GuidanceRequest(BaseModel):
    instrument_name: str
    question: str
    context: str | None = None
    mode: Literal["ai", "module"] = "ai"


class GuidanceResponse(BaseModel):
    instrument_name: str
    answer: str
    model_used: str


class MetricsResponse(BaseModel):
    events_observed: int
    mean_latency_ms: float
    mean_fps: float
    mean_confidence: float
    confidence_stability: float
    latest_timestamp: datetime | None
