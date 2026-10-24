"""Router do domínio `inference`: handlers das rotas versionadas ``/api/vN`` deste domínio."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ...config import get_settings
from ...database import get_db
from ...errors import invalid_input_as_422
from ...governance.audit import record_audit
from ...schemas import (
    BatchPredictionRequest,
    ExplainResponse,
    FeedbackRequest,
    NamedExplainRequest,
    NamedPredictionRequest,
    PredictionRequest,
    PredictionResponse,
)
from ...services import get_service

router = APIRouter()
settings = get_settings()
service = get_service()


@router.post(
    "/api/v1/predict", name="predict", response_model=PredictionResponse, tags=["inference"], deprecated=True
)
def predict_v1(payload: PredictionRequest, request: Request, db: Session = Depends(get_db)):
    return service.predict(
        db,
        payload.model_name or settings.default_model,
        payload.as_list(),
        request_id=request.state.request_id,
    )


@router.post("/api/v1/predict/batch", response_model=list[PredictionResponse], tags=["inference"])
def predict_batch(payload: BatchPredictionRequest, request: Request, db: Session = Depends(get_db)):
    if len(payload.items) > settings.max_batch_size:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"Batch exceeds max size of {settings.max_batch_size}"
        )
    items = [(item.model_name or settings.default_model, item.as_list()) for item in payload.items]
    return service.predict_batch(db, items, request_id=request.state.request_id)


@router.post("/api/v2/predictions/{prediction_id}/feedback", tags=["monitoring"])
def prediction_feedback(prediction_id: int, payload: FeedbackRequest, db: Session = Depends(get_db)) -> dict:
    try:
        log = service.record_feedback(db, prediction_id, payload.actual_class)
    except KeyError as exc:
        raise HTTPException(404, "Prediction not found") from exc
    record_audit(db, "prediction_feedback", prediction_id=prediction_id)
    return {
        "prediction_id": log.id,
        "predicted_class": log.predicted_class,
        "actual_class": log.actual_class,
        "correct": log.predicted_class == log.actual_class,
    }


@router.post(
    "/api/v2/explain",
    name="explain",
    response_model=ExplainResponse,
    tags=["explainability"],
    deprecated=True,
)
def explain_v2(payload: PredictionRequest) -> ExplainResponse:
    return ExplainResponse(**service.explain(payload.model_name or settings.default_model, payload.as_list()))


@router.post("/api/v8/predict", name="predict", tags=["inference"])
def predict_v8(payload: NamedPredictionRequest, request: Request, db: Session = Depends(get_db)) -> dict:
    """Predição com features nomeadas, para modelos de qualquer dataset.

    Se o modelo tem um canário ativo, o tráfego é dividido entre a versão estável e a
    candidata; a resposta diz qual braço (`arm`) atendeu.
    """
    return service.predict_routed(
        db,
        payload.model_name,
        payload.features,
        request_id=request.state.request_id,
        routing_key=payload.routing_key,
    )


@router.post("/api/v8/explain", name="explain", tags=["explainability"])
def explain_v8(payload: NamedExplainRequest) -> dict:
    """Shapley exato com features nomeadas (até 10 features)."""
    with invalid_input_as_422():
        vector = service.feature_vector(payload.model_name, payload.features)
        return service.explain(payload.model_name, vector)
