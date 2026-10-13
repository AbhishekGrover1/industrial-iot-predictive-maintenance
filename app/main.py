"""FastAPI application: prediction API and web console."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .predictor import Predictor
from .schemas import Diagnosis, Health, SensorReadings

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("predictive_maintenance")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
STATIC_DIR = BASE_DIR / "static"
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and fail fast if the artefacts are unusable."""
    app.state.predictor = Predictor.load(MODEL_DIR)
    logger.info("Model ready (%d classes) from %s", len(app.state.predictor.labels), MODEL_DIR)
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description="Multi-class fault diagnosis for industrial machines, served from an XGBoost model.",
    version="1.0.0",
    contact={"name": "Abhishek Grover", "url": "https://github.com/AbhishekGrover1"},
    license_info={"name": "MIT", "identifier": "MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def set_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


def get_predictor(request: Request) -> Predictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return predictor


@app.get("/health", response_model=Health, tags=["system"], summary="Service status")
def health(predictor: Predictor = Depends(get_predictor)) -> Health:
    return Health(status="ok", classes=predictor.labels)


@app.post("/predict", response_model=Diagnosis, tags=["inference"], summary="Diagnose a machine from sensor readings")
def predict(readings: SensorReadings, predictor: Predictor = Depends(get_predictor)) -> Diagnosis:
    try:
        diagnosis = predictor.predict(readings)
    except Exception:
        logger.exception("Inference failed for %s", readings.model_dump())
        raise HTTPException(status_code=500, detail="Inference failed.") from None
    logger.info("type=%s -> %s (%.1f%%)", readings.type, diagnosis.prediction, diagnosis.confidence_score * 100)
    return diagnosis


# Registered last so the API routes above always take precedence.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="console")
