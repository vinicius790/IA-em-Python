"""Router do domínio `registry`: handlers das rotas versionadas ``/api/vN`` deste domínio."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...api_v4 import _record_or_404
from ...api_v8 import _canary_or_404
from ...config import get_settings
from ...database import get_db
from ...db_models import CanaryDeployment, ExperimentRun
from ...deployment.canary import CanaryRule
from ...deployment.manifest import build_manifest
from ...errors import CanaryConflictError, GateFailedError, invalid_input_as_422
from ...governance.audit import record_audit
from ...governance.rbac import Principal
from ...model_selection import leaderboard
from ...schemas import (
    CanaryRequest,
    CanaryTrafficRequest,
    ComparisonRequest,
    ExperimentRead,
    ModelInfo,
    PromotionRequest,
)
from ...services import get_service

router = APIRouter()
settings = get_settings()
service = get_service()


@router.get("/api/v1/models", tags=["machine-learning"])
def models() -> list[dict]:
    return service.list_models()


@router.get("/api/v1/models/{model_name}", tags=["machine-learning"])
def model(model_name: str) -> dict:
    record = service.latest_model(model_name)
    if not record:
        raise HTTPException(404, "Model not found")
    return record


@router.get("/api/v2/models/leaderboard", tags=["models"])
def model_leaderboard(metric: str = Query("f1_macro", min_length=1, max_length=64)) -> list[dict]:
    return leaderboard(service.list_models(), metric)


@router.post("/api/v2/models/{model_name}/promote", response_model=ModelInfo, tags=["models"])
def promote(model_name: str, payload: PromotionRequest, db: Session = Depends(get_db)) -> dict:
    try:
        return service.promote(
            db, model_name, payload.version, payload.stage, notes=payload.notes, force=payload.force
        )
    except KeyError as exc:
        raise HTTPException(404, f"Model {model_name} version {payload.version} not found") from exc
    except GateFailedError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {"message": "Release gate failed; use force to override", "gate": exc.gate},
        ) from exc


@router.get("/api/v2/experiments", response_model=list[ExperimentRead], tags=["experiments"])
def experiments(limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)):
    return list(db.scalars(select(ExperimentRun).order_by(ExperimentRun.id.desc()).limit(limit)).all())


@router.get("/api/v4/deployment/manifest", tags=["deployment"])
def deployment_manifest(model_name: str = "xgboost", version: int | None = Query(None, ge=1)) -> dict:
    record = _record_or_404(model_name, version)
    return build_manifest(record, str(service.registry.artifact_file(record)), settings.environment)


@router.post("/api/v8/comparisons", status_code=status.HTTP_201_CREATED, tags=["comparison"])
def create_comparison(payload: ComparisonRequest, db: Session = Depends(get_db)) -> dict:
    """Validação cruzada estratificada dos modelos pedidos, nas mesmas dobras, com intervalo
    de confiança t; grava um relatório JSON e um HTML por execução."""
    with invalid_input_as_422():
        return service.compare(
            db,
            payload.models,
            dataset=payload.dataset,
            folds=payload.folds,
            confidence=payload.confidence,
            random_state=payload.random_state,
        )


@router.get("/api/v8/comparisons", tags=["comparison"])
def comparisons(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    return service.list_comparisons(limit)


@router.get("/api/v8/comparisons/{run_id}", tags=["comparison"])
def comparison(run_id: str) -> dict:
    path = service.comparison_file(run_id, "json")
    if path is None:
        raise HTTPException(404, "Comparison run not found")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/api/v8/comparisons/{run_id}/html", response_class=HTMLResponse, tags=["comparison"])
def comparison_html(run_id: str) -> HTMLResponse:
    path = service.comparison_file(run_id, "html")
    if path is None:
        raise HTTPException(404, "Comparison run not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@router.post("/api/v8/canary", status_code=status.HTTP_201_CREATED, tags=["canary"])
def start_canary(payload: CanaryRequest, request: Request, db: Session = Depends(get_db)) -> dict:
    """Abre um canário: `traffic_percent`% das predições de `/api/v8/predict` vão para a candidata."""
    principal: Principal = request.state.principal
    try:
        with invalid_input_as_422():
            rule = CanaryRule(**payload.rule.model_dump())
        canary = service.start_canary(
            db,
            payload.model_name,
            payload.candidate_version,
            payload.traffic_percent,
            rule,
            created_by=principal.name,
        )
    except KeyError as exc:
        raise HTTPException(404, "Candidate version not found in the registry") from exc
    except CanaryConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return service.canary_report(db, canary)


@router.get("/api/v8/canary", tags=["canary"])
def canaries(
    model_name: str | None = Query(None, max_length=128),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[dict]:
    query = select(CanaryDeployment).order_by(CanaryDeployment.id.desc()).limit(limit)
    if model_name:
        query = query.where(CanaryDeployment.model_name == model_name)
    return [
        {
            "id": row.id,
            "model_name": row.model_name,
            "status": row.status,
            "stable_version": row.stable_version,
            "candidate_version": row.candidate_version,
            "traffic_percent": row.traffic_percent,
            "created_at": row.created_at,
            "decided_at": row.decided_at,
        }
        for row in db.scalars(query).all()
    ]


@router.get("/api/v8/canary/{canary_id}", tags=["canary"])
def canary_status(canary_id: int, db: Session = Depends(get_db)) -> dict:
    """Métricas separadas da estável e da candidata e o que a regra decidiria agora."""
    return service.canary_report(db, _canary_or_404(db, canary_id))


@router.post("/api/v8/canary/{canary_id}/evaluate", tags=["canary"])
def evaluate_canary(canary_id: int, db: Session = Depends(get_db)) -> dict:
    """Aplica a regra agora: promove, reverte ou mantém o canário."""
    try:
        return service.evaluate_canary(db, _canary_or_404(db, canary_id))
    except CanaryConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/api/v8/canary/{canary_id}/rollback", tags=["canary"])
def rollback_canary(canary_id: int, db: Session = Depends(get_db)) -> dict:
    """Reversão manual: encerra o canário e arquiva a candidata."""
    canary = _canary_or_404(db, canary_id)
    try:
        service.rollback_canary(db, canary, {"decision": "rollback", "reason": "manual rollback"})
    except CanaryConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return service.canary_report(db, canary)


@router.post("/api/v8/canary/{canary_id}/traffic", tags=["canary"])
def canary_traffic(canary_id: int, payload: CanaryTrafficRequest, db: Session = Depends(get_db)) -> dict:
    """Muda o percentual de tráfego da candidata de um canário ativo."""
    canary = _canary_or_404(db, canary_id)
    if canary.status != "active":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Canary {canary.id} is already {canary.status}")
    canary.traffic_percent = payload.traffic_percent
    db.commit()
    record_audit(db, "canary_traffic", canary_id=canary.id, traffic_percent=payload.traffic_percent)
    return service.canary_report(db, canary)
