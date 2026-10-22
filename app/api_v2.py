"""API v2: MLOps (jobs assíncronos, promoção com gate, monitoramento e explicabilidade)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .api_v1 import ensure_known_model, training_response
from .config import get_settings
from .data import TARGET_COLUMN, load_iris_dataframe
from .database import SessionLocal, database_ok, get_db
from .db_models import ExperimentRun, JobRecord, MonitoringReport
from .errors import GateFailedError
from .features.engineering import enrich_iris, feature_manifest
from .governance.audit import record_audit
from .integrations import optional_stack_status
from .jobs import job_manager
from .model_selection import leaderboard
from .observability import ACTIVE_JOBS
from .schemas import (
    AsyncTrainingResponse,
    DataQualityResponse,
    DriftResponse,
    ExperimentRead,
    ExplainResponse,
    FeedbackRequest,
    JobRead,
    ModelInfo,
    MonitoringReportRead,
    PlatformStatus,
    PredictionRequest,
    PromotionRequest,
    TrainingRequest,
)
from .services import get_service

router = APIRouter(prefix="/api/v2")
settings = get_settings()
service = get_service()


@router.get("/platform", response_model=PlatformStatus, tags=["platform"])
def platform_status() -> PlatformStatus:
    return PlatformStatus(
        version=settings.app_version,
        environment=settings.environment,
        auth_required=bool(settings.api_key),
        model_count=len(service.list_models()),
        database=database_ok(),
        tracking_backend=service.tracker.backend,
        optional_stack=optional_stack_status(),
    )


@router.get("/data/quality", response_model=DataQualityResponse, tags=["data"])
def data_quality() -> DataQualityResponse:
    return DataQualityResponse(**service.quality_report())


@router.get("/features/manifest", tags=["data"])
def features_manifest(engineered: bool = False) -> dict:
    frame = load_iris_dataframe()
    return feature_manifest(enrich_iris(frame) if engineered else frame, target=TARGET_COLUMN)


@router.get("/models/leaderboard", tags=["models"])
def model_leaderboard(metric: str = Query("f1_macro", min_length=1, max_length=64)) -> list[dict]:
    return leaderboard(service.list_models(), metric)


@router.post("/models/{model_name}/promote", response_model=ModelInfo, tags=["models"])
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


def _run_training(request: TrainingRequest) -> dict:
    with SessionLocal() as db:
        run = service.train(
            db,
            request.model_name,
            request.test_size,
            request.cv_folds,
            request.tune_hyperparameters,
            request.optimization_trials,
            dataset=request.dataset,
        )
        return training_response(run).model_dump()


def _update_job_record(job_id: str, **values) -> None:
    with SessionLocal() as db:
        record = db.get(JobRecord, job_id)
        if record is not None:
            for key, value in values.items():
                setattr(record, key, value)
            db.commit()


@router.post("/training/async", response_model=AsyncTrainingResponse, status_code=202, tags=["training"])
async def async_training(payload: TrainingRequest) -> AsyncTrainingResponse:
    ensure_known_model(payload.model_name)
    job_ids: list[str] = []

    async def runner() -> dict:
        ACTIVE_JOBS.inc()
        _update_job_record(job_ids[0], status="running", started_at=datetime.now(UTC))
        try:
            result = await asyncio.to_thread(_run_training, payload)
        except Exception as exc:
            _update_job_record(
                job_ids[0], status="failed", error_message=str(exc), finished_at=datetime.now(UTC)
            )
            raise
        finally:
            ACTIVE_JOBS.dec()
        _update_job_record(
            job_ids[0],
            status="completed" if result["status"] == "completed" else "failed",
            result=result,
            error_message=result.get("error_message"),
            finished_at=datetime.now(UTC),
        )
        return result

    # O registro vai para o banco antes de o job começar a rodar (o runner só inicia
    # quando esta corrotina cede o controle), então o runner sempre o encontra.
    job = await job_manager.submit("model-training", runner)
    job_ids.append(job.id)
    with SessionLocal() as db:
        db.add(JobRecord(id=job.id, kind=job.kind, status=job.status, payload=payload.model_dump()))
        db.commit()
    return AsyncTrainingResponse(job_id=job.id, kind=job.kind, status=job.status, created_at=job.created_at)


@router.get("/jobs", response_model=list[JobRead], tags=["jobs"])
def jobs(limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)) -> list[dict]:
    records = db.scalars(select(JobRecord).order_by(JobRecord.created_at.desc()).limit(limit)).all()
    return [_job_dict(record) for record in records]


def _job_dict(record: JobRecord) -> dict:
    return {
        "id": record.id,
        "kind": record.kind,
        "status": record.status,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "result": record.result,
        "error": record.error_message,
    }


@router.get("/jobs/{job_id}", response_model=JobRead, tags=["jobs"])
def job(job_id: str, db: Session = Depends(get_db)) -> dict:
    record = db.get(JobRecord, job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    return _job_dict(record)


@router.get("/experiments", response_model=list[ExperimentRead], tags=["experiments"])
def experiments(limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)):
    return list(db.scalars(select(ExperimentRun).order_by(ExperimentRun.id.desc()).limit(limit)).all())


@router.get("/monitoring/drift/{model_name}", response_model=DriftResponse, tags=["monitoring"])
def drift(model_name: str, db: Session = Depends(get_db)) -> DriftResponse:
    return DriftResponse(**service.drift_report(db, model_name))


@router.get("/monitoring/{model_name}", tags=["monitoring"])
def monitoring(model_name: str, db: Session = Depends(get_db)) -> dict:
    return service.monitoring_summary(db, model_name)


@router.post(
    "/monitoring/{model_name}/reports",
    response_model=MonitoringReportRead,
    status_code=201,
    tags=["monitoring"],
)
def create_monitoring_report(model_name: str, db: Session = Depends(get_db)):
    return service.snapshot_monitoring(db, model_name)


@router.get(
    "/monitoring/{model_name}/reports", response_model=list[MonitoringReportRead], tags=["monitoring"]
)
def monitoring_reports(model_name: str, limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    query = (
        select(MonitoringReport)
        .where(MonitoringReport.model_name == model_name)
        .order_by(MonitoringReport.id.desc())
        .limit(limit)
    )
    return list(db.scalars(query).all())


@router.post("/predictions/{prediction_id}/feedback", tags=["monitoring"])
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


@router.post("/explain", response_model=ExplainResponse, tags=["explainability"])
def explain(payload: PredictionRequest) -> ExplainResponse:
    return ExplainResponse(**service.explain(payload.model_name or settings.default_model, payload.as_list()))
