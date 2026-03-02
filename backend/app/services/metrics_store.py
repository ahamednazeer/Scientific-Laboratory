from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np


@dataclass
class MetricEvent:
    timestamp: datetime
    latency_ms: float
    fps_estimate: float
    confidences: list[float]


class MetricsTracker:
    def __init__(self, window_size: int = 500) -> None:
        self.window_size = window_size
        self.events: deque[MetricEvent] = deque(maxlen=window_size)

    def record(self, *, latency_ms: float, fps_estimate: float, confidences: list[float]) -> None:
        self.events.append(
            MetricEvent(
                # Keep metric timestamps UTC-aware for consistent downstream analytics.
                timestamp=datetime.now(UTC),
                latency_ms=latency_ms,
                fps_estimate=fps_estimate,
                confidences=confidences,
            )
        )

    def snapshot(self) -> dict:
        if not self.events:
            return {
                "events_observed": 0,
                "mean_latency_ms": 0.0,
                "mean_fps": 0.0,
                "mean_confidence": 0.0,
                "confidence_stability": 0.0,
                "latest_timestamp": None,
            }

        latencies = np.array([e.latency_ms for e in self.events], dtype=np.float32)
        fps = np.array([e.fps_estimate for e in self.events], dtype=np.float32)
        flattened_conf = np.array([c for e in self.events for c in e.confidences], dtype=np.float32)

        mean_conf = float(flattened_conf.mean()) if flattened_conf.size else 0.0
        std_conf = float(flattened_conf.std()) if flattened_conf.size else 0.0
        confidence_stability = float(max(0.0, 1.0 - std_conf))

        return {
            "events_observed": len(self.events),
            "mean_latency_ms": float(latencies.mean()),
            "mean_fps": float(fps.mean()),
            "mean_confidence": mean_conf,
            "confidence_stability": confidence_stability,
            "latest_timestamp": self.events[-1].timestamp,
        }
