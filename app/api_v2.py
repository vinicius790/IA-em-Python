"""Helpers da API v2 (handlers em app.api.routers): MLOps (jobs assíncronos, promoção com gate, monitoramento e explicabilidade)."""

from __future__ import annotations

from .api_v1 import training_response
from .config import get_settings
from .database import SessionLocal
from .db_models import JobRecord
from .schemas import (
    TrainingRequest,
)
from .services import get_service

settings = get_settings()
service = get_service()


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


VERSION = 2


def __getattr__(name: str):
    """Compatibilidade: ``router`` agora é montado a partir de ``app.api.routers``."""
    if name == "router":
        from .api import versioned_router

        return versioned_router(VERSION)
    raise AttributeError(name)
