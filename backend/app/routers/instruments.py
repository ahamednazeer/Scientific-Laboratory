from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Instrument
from app.schemas import InstrumentOut

router = APIRouter(prefix="/instruments", tags=["instruments"])


@router.get("", response_model=list[InstrumentOut])
def list_instruments(db: Session = Depends(get_db)):
    return db.query(Instrument).order_by(Instrument.name.asc()).all()


@router.get("/{instrument_id}", response_model=InstrumentOut)
def get_instrument(instrument_id: int, db: Session = Depends(get_db)):
    item = db.query(Instrument).filter(Instrument.id == instrument_id).one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Instrument not found")
    return item
