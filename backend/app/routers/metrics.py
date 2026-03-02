from fastapi import APIRouter, Request

from app.schemas import MetricsResponse

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("", response_model=MetricsResponse)
def get_metrics(request: Request):
    snapshot = request.app.state.metrics.snapshot()
    return MetricsResponse(**snapshot)
