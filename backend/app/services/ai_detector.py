from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

import httpx
import numpy as np
from PIL import Image

from app.core.config import Settings
from app.services.refinement import RawDetection


@dataclass
class ParsedAIDetection:
    label: str
    confidence: float
    normalized_bbox: tuple[float, float, float, float] | None = None


class AIDetector:
    """LLM-vision detector that maps scene content to known lab instrument labels."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.last_status: str = "idle"
        self.last_error: str | None = None
        self.last_model_used: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.settings.groq_api_key)

    def detect(
        self,
        *,
        image_b64: str,
        image: Image.Image,
        candidate_labels: list[str],
    ) -> list[RawDetection]:
        self.last_status = "idle"
        self.last_error = None
        self.last_model_used = None

        if not self.settings.groq_api_key:
            self.last_status = "ai-unavailable"
            self.last_error = "GROQ_API_KEY is missing."
            return []
        if not candidate_labels:
            self.last_status = "ai-empty"
            self.last_error = "No candidate instrument labels found in database."
            return []

        content = self._call_groq(image_b64=image_b64, candidate_labels=candidate_labels)
        if not content:
            if self.last_status != "ai-error":
                self.last_status = "ai-empty"
                self.last_error = "AI returned empty content."
            return []

        parsed = self._parse_json_payload(content)
        detections: list[ParsedAIDetection]
        if parsed is None:
            detections = self._extract_detections_from_text(content, candidate_labels)
        else:
            detections = self._extract_detections(parsed, candidate_labels)
        if not detections:
            self.last_status = "ai-empty"
            self.last_error = "AI response did not contain any mappable allowed labels."
            return []

        width, height = image.size
        self.last_status = "ai"
        self.last_error = None
        return self._attach_boxes(detections=detections, image=image, width=width, height=height)

    def _build_payload(self, *, model: str, image_b64: str, candidate_labels: list[str]) -> dict[str, Any]:
        allowed = ", ".join(candidate_labels)
        data_uri = image_b64 if image_b64.startswith("data:") else f"data:image/jpeg;base64,{image_b64}"
        max_items = self._max_ai_detections()

        prompt = (
            "Detect every visible laboratory instrument in this image, including illustrations/icons.\n"
            f"Preferred labels: {allowed}.\n"
            "If an object uses a synonym/variant term, map it to the closest preferred label.\n"
            f"Return all detections (up to {max_items}) as strict JSON with this shape:\n"
            '{"detections":[{"label":"preferred label","confidence":0.0-1.0,'
            '"bbox":{"x1":0-1,"y1":0-1,"x2":0-1,"y2":0-1}}]}\n'
            "bbox must be normalized to image size, with x2>x1 and y2>y1.\n"
            "Use tight object boxes around the instrument body only. Exclude nearby text."
        )

        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict computer-vision labeling assistant. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            "temperature": 0.0,
            "max_completion_tokens": 900,
            "response_format": {"type": "json_object"},
        }

    def _candidate_models(self) -> list[str]:
        models: list[str] = []
        primary = (self.settings.groq_vision_model or "").strip()
        if primary:
            models.append(primary)
        for item in self.settings.groq_vision_fallback_models:
            raw = (item or "").strip()
            if not raw:
                continue
            # Accept either list entries or accidental comma-separated single entries.
            chunks = [part.strip() for part in raw.split(",") if part.strip()]
            for candidate in chunks:
                if candidate not in models:
                    models.append(candidate)
        return models

    def _call_groq(self, *, image_b64: str, candidate_labels: list[str]) -> str | None:
        headers = {
            "Authorization": f"Bearer {self.settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        last_error_text: str | None = None
        for model in self._candidate_models():
            payload = self._build_payload(model=model, image_b64=image_b64, candidate_labels=candidate_labels)
            try:
                with httpx.Client(timeout=22.0) as client:
                    response = client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                if response.status_code >= 400:
                    error_text = self._extract_error_text(response)
                    last_error_text = f"{response.status_code} ({model}): {error_text}"
                    # Retry other models for 400/404 model-compatibility issues.
                    if response.status_code in (400, 404):
                        continue
                    self.last_status = "ai-error"
                    self.last_error = last_error_text
                    return None

                data = response.json()
                self.last_model_used = model
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
                return str(content)
            except Exception as exc:
                last_error_text = f"{type(exc).__name__} ({model}): {exc}"
                continue

        self.last_status = "ai-error"
        self.last_error = last_error_text or "Unknown AI provider error."
        return None

    def _extract_error_text(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if message:
                    return str(message)
            if isinstance(error, str):
                return error
            return response.text[:400]
        except Exception:
            return response.text[:400]

    def _parse_json_payload(self, text: str) -> dict[str, Any] | list[Any] | None:
        text = text.strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except Exception:
            pass

        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            chunk = fence_match.group(1)
            try:
                return json.loads(chunk)
            except Exception:
                pass

        for open_char, close_char in (("{", "}"), ("[", "]")):
            start = text.find(open_char)
            if start < 0:
                continue
            depth = 0
            for idx in range(start, len(text)):
                char = text[idx]
                if char == open_char:
                    depth += 1
                elif char == close_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : idx + 1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break

        return None

    def _extract_detections(
        self,
        payload: dict[str, Any] | list[Any],
        candidate_labels: list[str],
    ) -> list[ParsedAIDetection]:
        rows: list[Any]
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("detections"), list):
                rows = payload["detections"]
            else:
                rows = payload.get("items", []) if isinstance(payload.get("items"), list) else []
        else:
            rows = []

        label_map = {self._normalize(label): label for label in candidate_labels}
        extracted: dict[str, ParsedAIDetection] = {}

        for row in rows:
            if not isinstance(row, dict):
                continue

            raw_label = row.get("label") or row.get("instrument") or row.get("name")
            if not raw_label:
                continue

            label = self._match_label(str(raw_label), candidate_labels, label_map)
            if not label:
                continue

            raw_conf = row.get("confidence", row.get("score", 0.0))
            try:
                confidence = float(raw_conf)
            except Exception:
                confidence = 0.0
            if confidence > 1.0:
                confidence = confidence / 100.0
            confidence = float(max(0.0, min(1.0, confidence)))
            bbox = self._extract_normalized_bbox(row)

            existing = extracted.get(label)
            if existing is None:
                extracted[label] = ParsedAIDetection(
                    label=label,
                    confidence=confidence,
                    normalized_bbox=bbox,
                )
                continue

            if confidence > existing.confidence:
                extracted[label] = ParsedAIDetection(
                    label=label,
                    confidence=confidence,
                    normalized_bbox=bbox or existing.normalized_bbox,
                )
            elif existing.normalized_bbox is None and bbox is not None:
                existing.normalized_bbox = bbox

        ranked = sorted(extracted.values(), key=lambda item: item.confidence, reverse=True)
        return ranked[: self._max_ai_detections()]

    def _extract_detections_from_text(self, text: str, candidate_labels: list[str]) -> list[ParsedAIDetection]:
        normalized_text = self._normalize(text)
        extracted: dict[str, ParsedAIDetection] = {}
        label_map = {self._normalize(label): label for label in candidate_labels}

        for key, label in label_map.items():
            if key and key in normalized_text:
                # Plain-text fallback score when model doesn't return structured confidence.
                existing = extracted.get(label)
                if existing is None or 0.55 > existing.confidence:
                    extracted[label] = ParsedAIDetection(label=label, confidence=0.55)

        ranked = sorted(extracted.values(), key=lambda item: item.confidence, reverse=True)
        return ranked[: self._max_ai_detections()]

    def _attach_boxes(
        self,
        *,
        detections: list[ParsedAIDetection],
        image: Image.Image,
        width: int,
        height: int,
    ) -> list[RawDetection]:
        items: list[RawDetection] = []
        missing_count = sum(1 for det in detections if det.normalized_bbox is None)
        placeholder_boxes = self._placeholder_bboxes(count=missing_count, width=width, height=height)
        placeholder_idx = 0

        for det in detections:
            resolved_bbox = (
                self._denormalize_bbox(det.normalized_bbox, width=width, height=height)
                if det.normalized_bbox is not None
                else None
            )
            if resolved_bbox is None:
                if placeholder_idx < len(placeholder_boxes):
                    resolved_bbox = placeholder_boxes[placeholder_idx]
                else:
                    resolved_bbox = (0.0, 0.0, float(width), float(height))
                placeholder_idx += 1
            resolved_bbox = self._refine_bbox_to_foreground(image=image, bbox=resolved_bbox)
            items.append(
                RawDetection(
                    label=det.label,
                    confidence=det.confidence,
                    bbox=resolved_bbox,
                )
            )

        return items

    def _max_ai_detections(self) -> int:
        requested = int(self.settings.detector_max_detections)
        return max(1, min(120, requested))

    def _extract_normalized_bbox(self, row: dict[str, Any]) -> tuple[float, float, float, float] | None:
        for key in ("bbox", "box", "bounds", "rect"):
            parsed = self._parse_bbox_from_value(row.get(key))
            if parsed is not None:
                return parsed

        for keys in (("x1", "y1", "x2", "y2"), ("left", "top", "right", "bottom")):
            parsed = self._parse_bbox_from_mapping(row, keys=keys, xywh=False)
            if parsed is not None:
                return parsed

        for keys in (("x", "y", "w", "h"), ("left", "top", "width", "height")):
            parsed = self._parse_bbox_from_mapping(row, keys=keys, xywh=True)
            if parsed is not None:
                return parsed

        return None

    def _parse_bbox_from_value(self, value: Any) -> tuple[float, float, float, float] | None:
        if value is None:
            return None

        if isinstance(value, dict):
            parsed = self._parse_bbox_from_mapping(value, keys=("x1", "y1", "x2", "y2"), xywh=False)
            if parsed is not None:
                return parsed
            parsed = self._parse_bbox_from_mapping(value, keys=("left", "top", "right", "bottom"), xywh=False)
            if parsed is not None:
                return parsed
            return self._parse_bbox_from_mapping(value, keys=("x", "y", "w", "h"), xywh=True)

        if isinstance(value, (list, tuple)) and len(value) >= 4:
            x1 = self._coerce_float(value[0])
            y1 = self._coerce_float(value[1])
            x2 = self._coerce_float(value[2])
            y2 = self._coerce_float(value[3])
            return self._normalize_bbox(x1=x1, y1=y1, x2=x2, y2=y2)

        if isinstance(value, str):
            numbers = [self._coerce_float(chunk) for chunk in re.findall(r"-?\d+(?:\.\d+)?", value)]
            if len(numbers) >= 4 and all(item is not None for item in numbers[:4]):
                return self._normalize_bbox(
                    x1=numbers[0],
                    y1=numbers[1],
                    x2=numbers[2],
                    y2=numbers[3],
                )

        return None

    def _parse_bbox_from_mapping(
        self,
        source: dict[str, Any],
        *,
        keys: tuple[str, str, str, str],
        xywh: bool,
    ) -> tuple[float, float, float, float] | None:
        values = [self._coerce_float(source.get(key)) for key in keys]
        if any(value is None for value in values):
            return None

        x1, y1, v3, v4 = values  # type: ignore[misc]
        if xywh:
            x2 = x1 + v3
            y2 = y1 + v4
        else:
            x2, y2 = v3, v4
        return self._normalize_bbox(x1=x1, y1=y1, x2=x2, y2=y2)

    def _normalize_bbox(
        self,
        *,
        x1: float | None,
        y1: float | None,
        x2: float | None,
        y2: float | None,
    ) -> tuple[float, float, float, float] | None:
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return None

        minimum = min(x1, y1, x2, y2)
        maximum = max(x1, y1, x2, y2)
        if minimum >= 0.0 and maximum <= 1.0:
            scale = 1.0
        elif minimum >= 0.0 and maximum <= 100.0:
            scale = 100.0
        else:
            return None

        nx1 = x1 / scale
        ny1 = y1 / scale
        nx2 = x2 / scale
        ny2 = y2 / scale

        left, right = sorted((nx1, nx2))
        top, bottom = sorted((ny1, ny2))

        left = max(0.0, min(1.0, left))
        right = max(0.0, min(1.0, right))
        top = max(0.0, min(1.0, top))
        bottom = max(0.0, min(1.0, bottom))

        if right - left < 0.02:
            center_x = (left + right) / 2.0
            left = max(0.0, center_x - 0.01)
            right = min(1.0, center_x + 0.01)
        if bottom - top < 0.02:
            center_y = (top + bottom) / 2.0
            top = max(0.0, center_y - 0.01)
            bottom = min(1.0, center_y + 0.01)

        if right <= left or bottom <= top:
            return None

        return (left, top, right, bottom)

    def _denormalize_bbox(
        self,
        normalized_bbox: tuple[float, float, float, float],
        *,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float] | None:
        x1, y1, x2, y2 = normalized_bbox

        left = float(max(0.0, min(1.0, x1)) * width)
        top = float(max(0.0, min(1.0, y1)) * height)
        right = float(max(0.0, min(1.0, x2)) * width)
        bottom = float(max(0.0, min(1.0, y2)) * height)

        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    def _placeholder_bboxes(self, *, count: int, width: int, height: int) -> list[tuple[float, float, float, float]]:
        if count <= 0:
            return []

        cols = max(1, int(math.ceil(math.sqrt(count))))
        rows = max(1, int(math.ceil(count / cols)))
        cell_w = float(width) / float(cols)
        cell_h = float(height) / float(rows)
        margin_x = max(2.0, cell_w * 0.12)
        margin_y = max(2.0, cell_h * 0.12)

        boxes: list[tuple[float, float, float, float]] = []
        for idx in range(count):
            col = idx % cols
            row = idx // cols

            x1 = col * cell_w + margin_x
            y1 = row * cell_h + margin_y
            x2 = (col + 1) * cell_w - margin_x
            y2 = (row + 1) * cell_h - margin_y

            if x2 <= x1:
                x2 = min(float(width), x1 + max(4.0, cell_w * 0.5))
            if y2 <= y1:
                y2 = min(float(height), y1 + max(4.0, cell_h * 0.5))

            boxes.append((float(x1), float(y1), float(x2), float(y2)))

        return boxes

    def _coerce_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _refine_bbox_to_foreground(
        self,
        *,
        image: Image.Image,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        width, height = image.size
        x1, y1, x2, y2 = bbox
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(x1 + 1.0, min(float(width), x2))
        y2 = max(y1 + 1.0, min(float(height), y2))

        ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        if ix2 - ix1 < 18 or iy2 - iy1 < 18:
            return (x1, y1, x2, y2)

        crop = image.crop((ix1, iy1, ix2, iy2)).convert("RGB")
        crop_w, crop_h = crop.size
        if crop_w < 8 or crop_h < 8:
            return (x1, y1, x2, y2)

        scale_x = 1.0
        scale_y = 1.0
        max_dim = max(crop_w, crop_h)
        if max_dim > 256:
            resize_ratio = 256.0 / float(max_dim)
            resized_w = max(8, int(round(crop_w * resize_ratio)))
            resized_h = max(8, int(round(crop_h * resize_ratio)))
            crop = crop.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
            scale_x = float(crop_w) / float(resized_w)
            scale_y = float(crop_h) / float(resized_h)

        arr = np.asarray(crop, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return (x1, y1, x2, y2)

        foreground_mask = self._build_foreground_mask(arr)
        if foreground_mask is None:
            return (x1, y1, x2, y2)

        component_bbox = self._best_component_bbox(foreground_mask)
        if component_bbox is None:
            return (x1, y1, x2, y2)

        cx1, cy1, cx2, cy2 = component_bbox
        pad_x = max(1, int(round((cx2 - cx1 + 1) * 0.06)))
        pad_y = max(1, int(round((cy2 - cy1 + 1) * 0.06)))
        cx1 = max(0, cx1 - pad_x)
        cy1 = max(0, cy1 - pad_y)
        cx2 = min(foreground_mask.shape[1] - 1, cx2 + pad_x)
        cy2 = min(foreground_mask.shape[0] - 1, cy2 + pad_y)

        local_x1 = float(cx1 * scale_x)
        local_y1 = float(cy1 * scale_y)
        local_x2 = float((cx2 + 1) * scale_x)
        local_y2 = float((cy2 + 1) * scale_y)

        rx1 = max(0.0, min(float(width - 1), ix1 + local_x1))
        ry1 = max(0.0, min(float(height - 1), iy1 + local_y1))
        rx2 = max(rx1 + 1.0, min(float(width), ix1 + local_x2))
        ry2 = max(ry1 + 1.0, min(float(height), iy1 + local_y2))

        original_area = max(1.0, (x2 - x1) * (y2 - y1))
        refined_area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
        area_ratio = refined_area / original_area
        if area_ratio < 0.06 or area_ratio > 1.15:
            return (x1, y1, x2, y2)

        return (rx1, ry1, rx2, ry2)

    def _build_foreground_mask(self, image_arr: np.ndarray) -> np.ndarray | None:
        h, w, _ = image_arr.shape
        if h < 6 or w < 6:
            return None

        border = np.vstack(
            [
                image_arr[0, :, :],
                image_arr[h - 1, :, :],
                image_arr[:, 0, :],
                image_arr[:, w - 1, :],
            ]
        )
        bg_color = np.median(border, axis=0)
        distance = np.linalg.norm(image_arr - bg_color, axis=2)

        border_dist = np.concatenate(
            [
                distance[0, :],
                distance[h - 1, :],
                distance[:, 0],
                distance[:, w - 1],
            ]
        )
        border_noise = float(np.percentile(border_dist, 90))
        threshold = max(14.0, border_noise * 2.2)
        mask = distance > threshold

        mask_ratio = float(mask.mean())
        if mask_ratio < 0.01 or mask_ratio > 0.82:
            return None

        cleaned = self._remove_border_connected(mask)
        if cleaned is None:
            return None

        cleaned_ratio = float(cleaned.mean())
        if cleaned_ratio < 0.004 or cleaned_ratio > 0.7:
            return None

        return cleaned

    def _remove_border_connected(self, mask: np.ndarray) -> np.ndarray | None:
        work = mask.copy()
        h, w = work.shape
        q: deque[tuple[int, int]] = deque()

        for x in range(w):
            if work[0, x]:
                q.append((0, x))
            if work[h - 1, x]:
                q.append((h - 1, x))
        for y in range(h):
            if work[y, 0]:
                q.append((y, 0))
            if work[y, w - 1]:
                q.append((y, w - 1))

        while q:
            y, x = q.popleft()
            if y < 0 or y >= h or x < 0 or x >= w or not work[y, x]:
                continue
            work[y, x] = False
            q.append((y - 1, x))
            q.append((y + 1, x))
            q.append((y, x - 1))
            q.append((y, x + 1))

        if work.any():
            return work
        return mask if mask.any() else None

    def _best_component_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        h, w = mask.shape
        visited = np.zeros((h, w), dtype=bool)
        center_x = (w - 1) / 2.0
        center_y = (h - 1) / 2.0
        diagonal = math.sqrt((w * w) + (h * h))

        best_score = -1.0
        best_bbox: tuple[int, int, int, int] | None = None

        for y in range(h):
            for x in range(w):
                if not mask[y, x] or visited[y, x]:
                    continue
                q: deque[tuple[int, int]] = deque([(y, x)])
                visited[y, x] = True

                min_x = max_x = x
                min_y = max_y = y
                count = 0
                sum_x = 0.0
                sum_y = 0.0

                while q:
                    cy, cx = q.popleft()
                    count += 1
                    sum_x += cx
                    sum_y += cy
                    if cx < min_x:
                        min_x = cx
                    if cx > max_x:
                        max_x = cx
                    if cy < min_y:
                        min_y = cy
                    if cy > max_y:
                        max_y = cy

                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        q.append((ny, nx))

                if count < 20:
                    continue

                comp_center_x = sum_x / count
                comp_center_y = sum_y / count
                center_distance = math.sqrt((comp_center_x - center_x) ** 2 + (comp_center_y - center_y) ** 2)
                center_penalty = 1.0 + (center_distance / max(1.0, diagonal))
                score = count / center_penalty

                if score > best_score:
                    best_score = score
                    best_bbox = (min_x, min_y, max_x, max_y)

        return best_bbox

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _match_label(self, raw_label: str, candidate_labels: list[str], label_map: dict[str, str]) -> str | None:
        normalized = self._normalize(raw_label)
        if not normalized:
            return None

        if normalized in label_map:
            return label_map[normalized]

        alias_target = self._resolve_alias_target(normalized, label_map)
        if alias_target:
            return alias_target

        # Partial inclusion matching (both directions) for robust parsing.
        for key, label in label_map.items():
            if normalized in key or key in normalized:
                return label

        close = get_close_matches(normalized, list(label_map.keys()), n=1, cutoff=0.6)
        if close:
            return label_map[close[0]]

        # Final fallback: exact case-insensitive equality against candidate labels.
        for label in candidate_labels:
            if label.lower() == raw_label.lower():
                return label

        return None

    def _resolve_alias_target(self, normalized: str, label_map: dict[str, str]) -> str | None:
        aliases = {
            "lab balance": "Analytical Balance",
            "laboratory balance": "Laboratory Balance",
            "precision balance": "Analytical Balance",
            "weighing balance": "Laboratory Balance",
            "hotplate": "Hot Plate",
            "hot plate": "Hot Plate",
            "magnetic stirrer": "Hot Plate Stirrer",
            "hot plate stirrer": "Hot Plate Stirrer",
            "spring scale": "Spring Balance",
            "ph paper": "pH Strip",
            "ph strip": "pH Strip",
            "litmus strip": "pH Strip",
            "erlenmeyer flask": "Volumetric Flask",
            "conical flask": "Volumetric Flask",
        }

        for alias, target in aliases.items():
            alias_key = self._normalize(alias)
            if normalized == alias_key or alias_key in normalized or normalized in alias_key:
                target_key = self._normalize(target)
                if target_key in label_map:
                    return label_map[target_key]

                close = get_close_matches(target_key, list(label_map.keys()), n=1, cutoff=0.55)
                if close:
                    return label_map[close[0]]

        return None
