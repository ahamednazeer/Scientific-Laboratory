from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import GuidanceRequest, GuidanceResponse

router = APIRouter(prefix="/guidance", tags=["guidance"])


@router.post("/ask", response_model=GuidanceResponse)
async def ask_guidance(payload: GuidanceRequest, request: Request, db: Session = Depends(get_db)):
    service = request.app.state.guidance_service
    answer, model_used = await service.answer(
        instrument_name=payload.instrument_name,
        question=payload.question,
        context=payload.context,
        mode=payload.mode,
        db=db,
    )
    return GuidanceResponse(
        instrument_name=payload.instrument_name,
        answer=answer,
        model_used=model_used,
    )
