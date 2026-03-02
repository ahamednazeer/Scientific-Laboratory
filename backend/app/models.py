from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Instrument(Base):
    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    category: Mapped[str] = mapped_column(String(80), default="general")
    description: Mapped[str] = mapped_column(Text)
    operation_steps: Mapped[str] = mapped_column(Text)
    safety_warnings: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    fps_estimate: Mapped[float] = mapped_column(Float)
    avg_confidence: Mapped[float] = mapped_column(Float)
    confidence_std: Mapped[float] = mapped_column(Float)
    confidence_stability: Mapped[float] = mapped_column(Float)
    detected_labels: Mapped[str] = mapped_column(Text)
