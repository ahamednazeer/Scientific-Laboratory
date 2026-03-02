from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db import Base, SessionLocal, engine
from app.routers import detection, guidance, health, instruments, metrics
from app.services.ai_detector import AIDetector
from app.services.detector import DetectorPipeline
from app.services.guidance import GuidanceService
from app.services.instrument_catalog import seed_instruments_if_needed
from app.services.kalman import BBoxStabilizer
from app.services.metrics_store import MetricsTracker

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        seed_instruments_if_needed(db)
    finally:
        db.close()

    app.state.settings = settings
    app.state.detector_pipeline = DetectorPipeline(settings)
    app.state.ai_detector = AIDetector(settings)
    app.state.stabilizer = BBoxStabilizer()
    app.state.metrics = MetricsTracker(window_size=1000)
    app.state.guidance_service = GuidanceService(settings)
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Web-native AR backend for laboratory instrument recognition and instructional guidance. "
        "Supports EfficientDet integration, Kalman-based stabilization, context-aware NMS, and Groq-driven Q&A."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(instruments.router, prefix="/api/v1")
app.include_router(detection.router, prefix="/api/v1")
app.include_router(guidance.router, prefix="/api/v1")
app.include_router(metrics.router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
