from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import Instrument
from app.services.refinement import RawDetection


@dataclass
class DetectorState:
    model_name: str
    active_mode: str
    ready: bool
    detail: str


class MockDetector:
    model_name = "mock-efficientdet-simulator"

    def detect(self, image: Image.Image, labels: list[str]) -> list[RawDetection]:
        if not labels:
            return []

        width, height = image.size
        resized = image.convert("L").resize((64, 64))
        arr = np.asarray(resized, dtype=np.float32)

        edge_strength = np.abs(np.diff(arr, axis=1)).mean() / 255.0
        brightness = arr.mean() / 255.0

        digest = hashlib.sha1(arr.tobytes()).digest()
        num = 1 + (digest[0] % min(3, len(labels)))

        detections: list[RawDetection] = []
        for i in range(num):
            label = labels[digest[i + 1] % len(labels)]
            base_x = (digest[i + 5] / 255.0) * 0.7
            base_y = (digest[i + 9] / 255.0) * 0.7
            box_w = max(0.12, 0.2 + 0.2 * brightness)
            box_h = max(0.12, 0.2 + 0.2 * edge_strength)

            x1 = max(0.0, min(width * 0.9, base_x * width))
            y1 = max(0.0, min(height * 0.9, base_y * height))
            x2 = min(width, x1 + box_w * width)
            y2 = min(height, y1 + box_h * height)

            conf = float(min(0.96, 0.5 + 0.35 * brightness + 0.25 * edge_strength + (i * 0.03)))
            detections.append(RawDetection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))

        return detections


class EfficientDetDetector:
    """Loads a fine-tuned EfficientDet checkpoint and performs local inference."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.detector_arch

        self._available = False
        self._load_detail = "EfficientDet not initialized"
        self._loaded_num_classes: int | None = None
        self._failed_num_classes: int | None = None

        self._torch: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None
        self._model_input_size: int | None = None
        self._class_mismatch_note: str | None = None
        self._requested_label_classes: int | None = None

        self._label_map: list[str] | None = None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def detail(self) -> str:
        return self._load_detail

    def _resolve_device(self, torch_module: Any):
        desired = self.settings.detector_device
        if desired == "cuda":
            if not torch_module.cuda.is_available():
                raise RuntimeError("DETECTOR_DEVICE=cuda but CUDA is not available")
            return torch_module.device("cuda")
        if desired == "mps":
            if not torch_module.backends.mps.is_available():
                raise RuntimeError("DETECTOR_DEVICE=mps but MPS is not available")
            return torch_module.device("mps")
        if desired == "cpu":
            return torch_module.device("cpu")

        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")

    def _extract_state_dict(self, checkpoint: Any) -> dict[str, Any]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict_ema", "state_dict", "model"):
                inner = checkpoint.get(key)
                if isinstance(inner, dict):
                    checkpoint = inner
                    break

        if not isinstance(checkpoint, dict):
            raise RuntimeError("Checkpoint format is not a state_dict dictionary")

        normalized: dict[str, Any] = {}
        for key, value in checkpoint.items():
            candidate = key
            if candidate.startswith("module."):
                candidate = candidate[len("module.") :]
            normalized[candidate] = value

        return normalized

    def _infer_checkpoint_num_classes(self, state_dict: dict[str, Any]) -> int | None:
        class_head_key = None
        box_head_key = None
        for key in state_dict.keys():
            if key.endswith("class_net.predict.conv_pw.weight"):
                class_head_key = key
            elif key.endswith("box_net.predict.conv_pw.weight"):
                box_head_key = key

        if not class_head_key:
            return None

        class_shape = getattr(state_dict[class_head_key], "shape", None)
        if not class_shape or len(class_shape) == 0:
            return None
        class_channels = int(class_shape[0])

        anchors_per_location = None
        if box_head_key:
            box_shape = getattr(state_dict[box_head_key], "shape", None)
            if box_shape and len(box_shape) > 0:
                box_channels = int(box_shape[0])
                if box_channels % 4 == 0:
                    anchors_per_location = max(1, box_channels // 4)

        if anchors_per_location is None:
            anchors_per_location = 9  # EfficientDet defaults

        if class_channels % anchors_per_location != 0:
            return None

        return max(1, class_channels // anchors_per_location)

    def _align_state_dict_keys(self, *, state_dict: dict[str, Any], model_state_keys: set[str]) -> dict[str, Any]:
        if not model_state_keys:
            return state_dict

        sample_model_key = next(iter(model_state_keys))
        model_uses_prefix = sample_model_key.startswith("model.")
        ckpt_has_prefix = any(key.startswith("model.") for key in state_dict.keys())

        if model_uses_prefix and not ckpt_has_prefix:
            return {f"model.{key}": value for key, value in state_dict.items()}
        if not model_uses_prefix and ckpt_has_prefix:
            return {
                key[len("model.") :] if key.startswith("model.") else key: value
                for key, value in state_dict.items()
            }
        return state_dict

    def _resolve_input_size(self) -> int:
        requested = max(128, int(self.settings.detector_input_size))
        # BiFPN requires consistent pyramid scales; align to 128-step boundaries.
        return int(np.ceil(requested / 128.0) * 128)

    def _load_label_map(self, db_labels: list[str]) -> list[str]:
        if self._label_map is not None:
            return self._label_map

        label_map_path = Path(self.settings.detector_label_map_path)
        if label_map_path.exists():
            try:
                content = json.loads(label_map_path.read_text(encoding="utf-8"))
                if isinstance(content, list) and all(isinstance(item, str) for item in content):
                    labels = [item.strip() for item in content if item.strip()]
                    if labels:
                        self._label_map = labels
                        return self._label_map
            except Exception as exc:
                self._load_detail = f"Invalid label map JSON at {label_map_path}: {exc}"

        fallback = sorted({label.strip() for label in db_labels if label.strip()})
        self._label_map = fallback
        return self._label_map

    def _lazy_load(self, *, num_classes: int) -> None:
        target_classes = max(1, num_classes)
        if (
            self._available
            and self._model is not None
            and self._requested_label_classes == target_classes
        ):
            return
        if not self._available and self._failed_num_classes == target_classes and self._model is None:
            return

        weights_path = Path(self.settings.detector_weights_path)
        if not weights_path.exists():
            self._available = False
            self._load_detail = f"Missing checkpoint: {weights_path}"
            self._failed_num_classes = target_classes
            return

        try:
            import torch
            from effdet import create_model
        except Exception as exc:
            self._available = False
            self._load_detail = f"Missing EfficientDet runtime deps: {exc}"
            self._failed_num_classes = target_classes
            return

        try:
            device = self._resolve_device(torch)
            checkpoint = torch.load(weights_path, map_location=device)
            raw_state_dict = self._extract_state_dict(checkpoint)
            checkpoint_classes = self._infer_checkpoint_num_classes(raw_state_dict)
            model_num_classes = checkpoint_classes or target_classes

            input_size = self._resolve_input_size()
            model = create_model(
                self.settings.detector_arch,
                bench_task="predict",
                pretrained=False,
                pretrained_backbone=False,
                num_classes=model_num_classes,
                image_size=(input_size, input_size),
            )

            state_dict = self._align_state_dict_keys(
                state_dict=raw_state_dict,
                model_state_keys=set(model.state_dict().keys()),
            )
            load_result = model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()

            if self.settings.detector_use_fp16 and str(device).startswith("cuda"):
                model.half()

            self._torch = torch
            self._model = model
            self._device = device
            self._model_input_size = input_size
            self._available = True
            self._loaded_num_classes = model_num_classes
            self._requested_label_classes = target_classes
            self._failed_num_classes = None
            self._class_mismatch_note = None
            if checkpoint_classes and checkpoint_classes != target_classes:
                self._class_mismatch_note = (
                    f"Checkpoint classes={checkpoint_classes}, label_map classes={target_classes}. "
                    "Out-of-map classes will use fallback labels."
                )
            load_note = ""
            missing_keys = getattr(load_result, "missing_keys", [])
            unexpected_keys = getattr(load_result, "unexpected_keys", [])
            if missing_keys or unexpected_keys:
                load_note = (
                    f" missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
                )
            mismatch_note = f" {self._class_mismatch_note}" if self._class_mismatch_note else ""
            self._load_detail = (
                f"Loaded {self.settings.detector_arch} on {device} "
                f"(image_size={input_size}, classes={model_num_classes}).{load_note}{mismatch_note}"
            ).strip()
        except Exception as exc:
            self._available = False
            self._model = None
            self._torch = None
            self._device = None
            self._model_input_size = None
            self._loaded_num_classes = None
            self._requested_label_classes = None
            self._failed_num_classes = target_classes
            self._load_detail = f"Failed to load EfficientDet model: {exc}"

    def probe(self) -> None:
        labels = self._load_label_map([])
        if labels:
            self._lazy_load(num_classes=len(labels))

    def _prepare_input(self, image: Image.Image) -> tuple[Any, float, float]:
        assert self._torch is not None

        original_w, original_h = image.size
        input_size = self._model_input_size or self._resolve_input_size()

        img_scale_y = input_size / float(max(1, original_h))
        img_scale_x = input_size / float(max(1, original_w))
        img_scale = min(img_scale_x, img_scale_y)

        scaled_w = max(1, int(original_w * img_scale))
        scaled_h = max(1, int(original_h * img_scale))

        resized = image.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
        resized_arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = np.zeros((input_size, input_size, 3), dtype=np.float32)
        arr[:scaled_h, :scaled_w, :] = resized_arr

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        tensor = self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        if self.settings.detector_use_fp16 and str(self._device).startswith("cuda"):
            tensor = tensor.half()
        else:
            tensor = tensor.float()

        tensor = tensor.to(self._device)

        scale_x = 1.0 / img_scale
        scale_y = 1.0 / img_scale
        return tensor, scale_x, scale_y

    def _extract_prediction_rows(self, output: Any) -> np.ndarray:
        assert self._torch is not None

        candidate = output
        if isinstance(candidate, (list, tuple)) and candidate:
            candidate = candidate[0]

        if not self._torch.is_tensor(candidate):
            return np.empty((0, 6), dtype=np.float32)

        tensor = candidate.detach().to("cpu")
        if tensor.ndim == 3:
            tensor = tensor[0]

        if tensor.ndim != 2 or tensor.shape[-1] < 6:
            return np.empty((0, 6), dtype=np.float32)

        return tensor.numpy().astype(np.float32, copy=False)

    def _class_to_label(self, class_id: int, labels: list[str]) -> str | None:
        if self._class_mismatch_note:
            return f"class_{class_id}"

        if not labels:
            return f"class_{class_id}"

        adjusted = class_id - self.settings.detector_class_index_base
        if 0 <= adjusted < len(labels):
            return labels[adjusted]

        if 0 <= class_id < len(labels):
            return labels[class_id]
        if 1 <= class_id <= len(labels):
            return labels[class_id - 1]

        return f"class_{class_id}"

    def _decode_bbox(
        self,
        *,
        raw_box: tuple[float, float, float, float],
        width: int,
        height: int,
        scale_x: float,
        scale_y: float,
    ) -> tuple[float, float, float, float]:
        x1, y1, x3, y3 = raw_box

        if x3 > x1 and y3 > y1:
            x2, y2 = x3, y3
        else:
            x2, y2 = x1 + max(1.0, x3), y1 + max(1.0, y3)

        x1 *= scale_x
        y1 *= scale_y
        x2 *= scale_x
        y2 *= scale_y

        x1 = float(np.clip(x1, 0.0, max(1.0, width - 1.0)))
        y1 = float(np.clip(y1, 0.0, max(1.0, height - 1.0)))
        x2 = float(np.clip(x2, x1 + 1.0, float(width)))
        y2 = float(np.clip(y2, y1 + 1.0, float(height)))
        return x1, y1, x2, y2

    def detect(self, image: Image.Image, labels: list[str]) -> list[RawDetection]:
        label_map = self._load_label_map(labels)
        self._lazy_load(num_classes=len(label_map))

        if not self._available or self._model is None or self._torch is None:
            return []

        try:
            width, height = image.size
            input_tensor, scale_x, scale_y = self._prepare_input(image)

            with self._torch.no_grad():
                output = self._model(input_tensor)
        except Exception as exc:
            self._available = False
            self._load_detail = f"Inference failed: {exc}"
            return []

        rows = self._extract_prediction_rows(output)
        if rows.shape[0] == 0:
            return []

        max_detections = max(1, int(self.settings.detector_max_detections))
        rows = rows[:max_detections]

        detections: list[RawDetection] = []
        for row in rows:
            score = float(row[4])
            if score <= 0.0:
                continue

            class_id = int(round(float(row[5])))
            label = self._class_to_label(class_id, label_map)
            if label is None:
                continue

            bbox = self._decode_bbox(
                raw_box=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                width=width,
                height=height,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            detections.append(RawDetection(label=label, confidence=score, bbox=bbox))

        return detections


class DetectorPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mock = MockDetector()
        self.efficientdet = EfficientDetDetector(settings)

    @property
    def state(self) -> DetectorState:
        if self.settings.detector_mode == "efficientdet":
            self.efficientdet.probe()
            return DetectorState(
                model_name=self.efficientdet.model_name,
                active_mode="efficientdet",
                ready=self.efficientdet.available,
                detail=self.efficientdet.detail,
            )

        return DetectorState(
            model_name=self.mock.model_name,
            active_mode="mock",
            ready=True,
            detail="Mock detector active",
        )

    def detect(self, image: Image.Image, db: Session) -> list[RawDetection]:
        labels = [name for (name,) in db.query(Instrument.name).order_by(Instrument.name.asc()).all()]

        if self.settings.detector_mode == "efficientdet":
            detections = self.efficientdet.detect(image, labels)
            if self.efficientdet.available:
                filtered = [d for d in detections if d.confidence >= self.settings.detection_confidence_threshold]
                if filtered:
                    return filtered
                ranked = sorted(detections, key=lambda item: item.confidence, reverse=True)
                relaxed_threshold = min(0.001, self.settings.detection_confidence_threshold)
                relaxed = [d for d in ranked if d.confidence >= relaxed_threshold]
                return (relaxed or ranked)[:5]
            return []

        detections = self.mock.detect(image, labels)
        return [d for d in detections if d.confidence >= self.settings.detection_confidence_threshold]


def decode_base64_image(image_b64: str) -> Image.Image:
    payload = image_b64
    if image_b64.startswith("data:"):
        payload = image_b64.split(",", 1)[1]

    binary = base64.b64decode(payload)
    image = Image.open(io.BytesIO(binary)).convert("RGB")
    return image
