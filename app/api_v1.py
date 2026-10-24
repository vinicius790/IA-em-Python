"""Helpers da API v1 (handlers em app.api.routers): saúde, dados, treino síncrono, inferência e consulta ao registry."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.orm import Session

from .config import get_settings
from .database import database_ok
from .db_models import IrisRecord, TrainingRun
from .ml import available_models
from .schemas import (
    HealthResponse,
    ReadinessResponse,
    TrainingResponse,
)
from .services import get_service

# Rotas públicas (sem API key): sondas de saúde, métricas e o atalho para o console.
public_router = APIRouter(prefix="/api/v1")
settings = get_settings()
service = get_service()


@public_router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.app_version, environment=settings.environment)


@public_router.get("/ready", response_model=ReadinessResponse, tags=["system"])
def readiness() -> ReadinessResponse:
    db_ok = database_ok()
    model_available = service.latest_model(settings.default_model) is not None
    ready = db_ok and model_available
    return ReadinessResponse(
        status="ready" if ready else "not_ready", database=db_ok, model_available=model_available
    )


@public_router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@public_router.get("/dashboard", include_in_schema=False)
def dashboard() -> RedirectResponse:
    return RedirectResponse("/console/")


def _iris_or_404(db: Session, iris_id: int) -> IrisRecord:
    row = db.get(IrisRecord, iris_id)
    if row is None:
        raise HTTPException(404, "Iris record not found")
    return row


def ensure_known_model(model_name: str) -> None:
    if model_name not in available_models():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Unknown model '{model_name}'. Available: {', '.join(available_models())}",
        )


def training_response(run: TrainingRun) -> TrainingResponse:
    return TrainingResponse(
        run_id=run.id,
        model_name=run.model_name,
        status=run.status,
        metrics=run.metrics,
        artifact_path=run.artifact_path,
        dataset_hash=run.dataset_hash,
        lineage_hash=run.lineage_hash,
        model_card_path=run.model_card_path,
        error_message=run.error_message,
    )


VERSION = 1


def __getattr__(name: str):
    """Compatibilidade: ``router`` agora é montado a partir de ``app.api.routers``."""
    if name == "router":
        from .api import versioned_router

        return versioned_router(VERSION)
    raise AttributeError(name)
