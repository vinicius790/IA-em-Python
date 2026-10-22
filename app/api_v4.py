"""API v4: pipelines, auditoria, governança e análises do dataset."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from .analytics.statistics import correlations
from .api_v1 import ensure_known_model
from .config import get_settings
from .data import TARGET_COLUMN, load_iris_dataframe
from .database import get_db
from .db_models import PipelineRun
from .deployment.manifest import build_manifest
from .features.engineering import enrich_iris
from .features.selection import select_k_best, variance_filter
from .governance.audit import recent_audit
from .monitoring.anomaly import detect_anomalies
from .pipelines.executable import dry_run
from .pipelines.service import TrainingPipeline
from .schemas import DryRunRequest, PipelineRequest
from .services import get_service

router = APIRouter(prefix="/api/v4")
settings = get_settings()
service = get_service()


def _record_or_404(model_name: str, version: int | None) -> dict:
    record = service.registry.get(model_name, version) if version else service.latest_model(model_name)
    if record is None:
        raise HTTPException(404, "Model version not found in the registry")
    return record


@router.get("/data/anomalies", tags=["data"])
def anomalies() -> dict:
    return detect_anomalies(load_iris_dataframe())


@router.get("/features/selection", tags=["features"])
def selection(k: int = Query(4, ge=1, le=20), method: Literal["anova", "mutual_info"] = "anova") -> dict:
    frame = enrich_iris(load_iris_dataframe())
    return {"variance": variance_filter(frame), "best": select_k_best(frame, TARGET_COLUMN, k, method)}


@router.post("/pipelines/dry-run", tags=["pipelines"])
def pipeline_dry_run(payload: DryRunRequest | None = None) -> dict:
    """Executa o DAG de preparação (ingestão, qualidade, features, anomalias) sem treinar."""
    model_name = (payload or DryRunRequest()).model_name
    ensure_known_model(model_name)
    return dry_run(model_name)


@router.post("/pipelines/train", tags=["pipelines"])
def pipeline_train(payload: PipelineRequest | None = None, db: Session = Depends(get_db)) -> dict:
    request = payload or PipelineRequest()
    ensure_known_model(request.model_name)
    return TrainingPipeline(service).run(
        db, request.model_name, request.test_size, request.cv_folds, request.tune, request.trials
    )


@router.get("/pipelines/runs", tags=["pipelines"])
def pipeline_runs(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)) -> list[dict]:
    rows = db.scalars(select(PipelineRun).order_by(PipelineRun.id.desc()).limit(limit)).all()
    return [
        {
            "id": row.id,
            "pipeline_name": row.pipeline_name,
            "status": row.status,
            "model_name": row.model_name,
            "payload": row.payload,
            "created_at": row.created_at,
        }
        for row in rows
    ]


@router.get("/audit", tags=["governance"])
def audit_events(limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)) -> list[dict]:
    return recent_audit(db, limit)


@router.get("/governance/gate/{model_name}", tags=["governance"])
def release_gate(model_name: str, version: int | None = Query(None, ge=1)) -> dict:
    """Avalia o release gate sobre as métricas registradas de uma versão do modelo."""
    record = _record_or_404(model_name, version)
    return {"model": record["alias"], "stage": record["stage"], "gate": service.release_gate(record)}


@router.get("/analytics/correlations", tags=["analytics"])
def analytics_correlations(method: Literal["pearson", "spearman", "kendall"] = "pearson") -> dict:
    return correlations(load_iris_dataframe(), method)


@router.get("/deployment/manifest", tags=["deployment"])
def deployment_manifest(model_name: str = "xgboost", version: int | None = Query(None, ge=1)) -> dict:
    record = _record_or_404(model_name, version)
    return build_manifest(record, str(service.registry.artifact_file(record)), settings.environment)
