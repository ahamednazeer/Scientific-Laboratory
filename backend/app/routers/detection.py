from datetime import UTC, datetime
from time import perf_counter

import numpy as np
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import DetectionEvent, Instrument
from app.schemas import DetectionBox, DetectionOut, DetectionRequest, DetectionResponse
from app.services.detector import decode_base64_image
from app.services.kalman import BBoxStabilizer
from app.services.refinement import RawDetection, context_aware_nms, non_max_suppression

router = APIRouter(prefix="/detection", tags=["detection"])


def _to_box(bbox: tuple[float, float, float, float]) -> DetectionBox:
    x1, y1, x2, y2 = bbox
    return DetectionBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))


@router.post("/predict", response_model=DetectionResponse)
def predict(payload: DetectionRequest, request: Request, db: Session = Depends(get_db)):
    start = perf_counter()

    image = decode_base64_image(payload.image_b64)
    detector_pipeline = request.app.state.detector_pipeline
    ai_detector = request.app.state.ai_detector
    metrics = request.app.state.metrics

    instruments = db.query(Instrument).all()
    known_labels = [item.name for item in instruments]

    raw: list[RawDetection] = []
    source_used = "model"
    source_note: str | None = None
    if payload.mode == "ai":
        raw = ai_detector.detect(
            image_b64=payload.image_b64,
            image=image,
            candidate_labels=known_labels,
        )
        if raw:
            source_used = "ai"
            source_note = None
        else:
            source_used = ai_detector.last_status if ai_detector.last_status != "idle" else (
                "ai-unavailable" if not ai_detector.available else "ai-empty"
            )
            source_note = ai_detector.last_error
    else:
        raw = detector_pipeline.detect(image, db)
        source_used = "model"
        source_note = None

    compatible_pairs = {
        ("Test Tube Rack", "Pipette"),
        ("Volumetric Flask", "Pipette"),
        ("Microscope", "Test Tube Rack"),
    }

    refined: list[RawDetection]
    if payload.run_refinement:
        if source_used.startswith("ai"):
            # AI boxes can overlap for adjacent classes; keep cross-class labels and only suppress duplicates.
            refined = non_max_suppression(
                raw,
                iou_threshold=request.app.state.settings.nms_iou_threshold,
            )
        else:
            refined = context_aware_nms(
                raw,
                iou_threshold=request.app.state.settings.nms_iou_threshold,
                compatible_pairs=compatible_pairs,
            )
    else:
        refined = raw

    stabilizer: BBoxStabilizer = request.app.state.stabilizer

    label_to_instrument = {item.name: item for item in instruments}
    detections_out: list[DetectionOut] = []

    for det in refined:
        smoothed = stabilizer.smooth_detection(det.label, det.bbox)
        instrument = label_to_instrument.get(det.label)

        detections_out.append(
            DetectionOut(
                label=det.label,
                confidence=float(det.confidence),
                bbox=_to_box(det.bbox),
                smoothed_bbox=_to_box(smoothed.bbox),
                stability_score=smoothed.stability,
                instrument_id=instrument.id if instrument else None,
                description=instrument.description if instrument else None,
                operation_steps=instrument.operation_steps if instrument else None,
                safety_warnings=instrument.safety_warnings if instrument else None,
            )
        )

    latency_ms = (perf_counter() - start) * 1000
    fps_estimate = 1000.0 / max(1.0, latency_ms)

    confidences = [d.confidence for d in detections_out]
    conf_mean = float(np.mean(confidences)) if confidences else 0.0
    conf_std = float(np.std(confidences)) if confidences else 0.0
    conf_stability = float(max(0.0, 1.0 - conf_std))

    db.add(
        DetectionEvent(
            latency_ms=latency_ms,
            fps_estimate=fps_estimate,
            avg_confidence=conf_mean,
            confidence_std=conf_std,
            confidence_stability=conf_stability,
            detected_labels=", ".join([d.label for d in detections_out]),
        )
    )
    db.commit()

    metrics.record(latency_ms=latency_ms, fps_estimate=fps_estimate, confidences=confidences)

    return DetectionResponse(
        timestamp=datetime.now(UTC),
        latency_ms=latency_ms,
        fps_estimate=fps_estimate,
        detection_source_used=source_used,
        detection_source_note=source_note,
        detections=detections_out,
    )
