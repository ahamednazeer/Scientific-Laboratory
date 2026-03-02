from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Scientific Laboratory AR API"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    database_url: str = "sqlite:///./lab_ar.db"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"])

    detector_mode: Literal["mock", "efficientdet"] = "mock"
    detector_arch: str = "tf_efficientdet_d2"
    detector_weights_path: str = "models/efficientdet_lab_instruments.pth"
    detector_label_map_path: str = "models/class_labels.json"
    detector_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    detector_input_size: int = 768
    detector_max_detections: int = 50
    detector_class_index_base: int = 1
    detector_use_fp16: bool = False
    detection_confidence_threshold: float = 0.35
    nms_iou_threshold: float = 0.45

    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_vision_fallback_models: list[str] = Field(
        default_factory=lambda: [
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "llama-3.2-90b-vision-preview",
        ]
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
