from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanResult:
    bbox: tuple[float, float, float, float]
    stability: float


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over (x, y, w, h)."""

    def __init__(self, bbox: tuple[float, float, float, float], dt: float = 1.0) -> None:
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        self.x = np.array([x1, y1, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 20.0

        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = dt

        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.Q = np.eye(8, dtype=np.float32) * 0.08
        self.R = np.eye(4, dtype=np.float32) * 1.5

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, bbox: tuple[float, float, float, float]) -> KalmanResult:
        x1, y1, x2, y2 = bbox
        z = np.array([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=np.float32)

        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)

        self.x = self.x + (k @ y)
        ident = np.eye(8, dtype=np.float32)
        self.P = (ident - k @ self.H) @ self.P

        smoothed = self.to_bbox()
        stability = float(np.clip(1.0 - (np.trace(self.P[:4, :4]) / 300.0), 0.0, 1.0))
        return KalmanResult(bbox=smoothed, stability=stability)

    def to_bbox(self) -> tuple[float, float, float, float]:
        x1, y1, w, h = self.x[:4]
        return (float(x1), float(y1), float(x1 + max(1.0, w)), float(y1 + max(1.0, h)))


class BBoxStabilizer:
    def __init__(self) -> None:
        self.filters: dict[str, KalmanBoxFilter] = {}

    def smooth_detection(self, label: str, bbox: tuple[float, float, float, float]) -> KalmanResult:
        if label not in self.filters:
            self.filters[label] = KalmanBoxFilter(bbox)

        tracker = self.filters[label]
        tracker.predict()
        return tracker.update(bbox)
