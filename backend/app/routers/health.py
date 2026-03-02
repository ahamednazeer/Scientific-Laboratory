from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(request: Request):
    pipeline = request.app.state.detector_pipeline
    ai_detector = request.app.state.ai_detector
    return {
        "status": "ok",
        "detector_mode": pipeline.state.active_mode,
        "model_name": pipeline.state.model_name,
        "detector_ready": pipeline.state.ready,
        "detector_detail": pipeline.state.detail,
        "ai_detection_available": ai_detector.available,
    }
