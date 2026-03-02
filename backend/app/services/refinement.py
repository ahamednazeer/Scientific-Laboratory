from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class RawDetection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / union)


def non_max_suppression(detections: Sequence[RawDetection], iou_threshold: float) -> list[RawDetection]:
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    selected: list[RawDetection] = []

    while ordered:
        current = ordered.pop(0)
        selected.append(current)

        remaining: list[RawDetection] = []
        for candidate in ordered:
            same_label = candidate.label == current.label
            overlap = iou(candidate.bbox, current.bbox)
            if same_label and overlap > iou_threshold:
                continue
            remaining.append(candidate)
        ordered = remaining

    return selected


def context_aware_nms(
    detections: Sequence[RawDetection],
    *,
    iou_threshold: float,
    compatible_pairs: set[tuple[str, str]] | None = None,
) -> list[RawDetection]:
    """Applies standard NMS and suppresses implausible overlapping cross-class detections."""
    pruned = non_max_suppression(detections, iou_threshold)
    compatible_pairs = compatible_pairs or set()

    output: list[RawDetection] = []
    for det in sorted(pruned, key=lambda d: d.confidence, reverse=True):
        conflict = False
        for kept in output:
            overlap = iou(det.bbox, kept.bbox)
            pair = (det.label, kept.label)
            reverse_pair = (kept.label, det.label)
            compatible = pair in compatible_pairs or reverse_pair in compatible_pairs
            if overlap > 0.55 and det.label != kept.label and not compatible:
                conflict = True
                break
        if not conflict:
            output.append(det)

    return output
