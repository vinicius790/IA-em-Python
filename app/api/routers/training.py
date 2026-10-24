"""Router do domínio `training`: handlers das rotas versionadas ``/api/vN`` deste domínio."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...api_v1 import ensure_known_model, training_response
from ...api_v2 import _run_training, _update_job_record
from ...config import get_settings
from ...data import load_iris_dataframe
from ...database import SessionLocal, get_db
from ...db_models import JobRecord, PipelineRun, TrainingRun
from ...errors import invalid_input_as_422
from ...jobs import job_manager
from ...ml import available_models
from ...model_selection import pareto_frontier
from ...modeling.evaluation import bootstrap_metric
from ...modeling.robustness import perturbation_report
from ...observability import ACTIVE_JOBS
from ...pipelines.executable import dry_run
from ...pipelines.service import TrainingPipeline
from ...research.experiment_matrix import ExperimentMatrix, ExperimentSpec
from ...schemas import (
    AsyncTrainingResponse,
    DryRunRequest,
    EvaluatePredictionsRequest,
    ExperimentRequest,
    PipelineRequest,
    SearchRequest,
    TrainingRequest,
    TrainingResponse,
    TrainingRunRead,
    check_cv_budget,
)
from ...search.ingest import documentation_index
from ...services import get_service

router = APIRouter()
settings = get_settings()
service = get_service()


@router.post("/api/v1/training/train", response_model=TrainingResponse, tags=["machine-learning"])
def train(payload: TrainingRequest, db: Session = Depends(get_db)) -> TrainingResponse:
    ensure_known_model(payload.model_name)
    run = service.train(
        db,
        payload.model_name,
        payload.test_size,
        payload.cv_folds,
        payload.tune_hyperparameters,
        payload.optimization_trials,
        dataset=payload.dataset,
    )
    if run.status != "completed":
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Training run {run.id} failed: {run.error_message}"
        )
    return training_response(run)


@router.get("/api/v1/training/runs", response_model=list[TrainingRunRead], tags=["machine-learning"])
def training_runs(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    return list(db.scalars(select(TrainingRun).order_by(TrainingRun.id.desc()).limit(limit)).all())


@router.post(
    "/api/v2/training/async", response_model=AsyncTrainingResponse, status_code=202, tags=["training"]
)
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


@router.post("/api/v4/pipelines/dry-run", tags=["pipelines"])
def pipeline_dry_run(payload: DryRunRequest | None = None) -> dict:
    """Executa o DAG de preparação (ingestão, qualidade, features, anomalias) sem treinar."""
    model_name = (payload or DryRunRequest()).model_name
    ensure_known_model(model_name)
    return dry_run(model_name)


@router.post("/api/v4/pipelines/train", tags=["pipelines"])
def pipeline_train(payload: PipelineRequest | None = None, db: Session = Depends(get_db)) -> dict:
    request = payload or PipelineRequest()
    ensure_known_model(request.model_name)
    return TrainingPipeline(service).run(
        db, request.model_name, request.test_size, request.cv_folds, request.tune, request.trials
    )


@router.get("/api/v4/pipelines/runs", tags=["pipelines"])
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


@router.get("/api/v5/studio/frontier", tags=["studio"])
def frontier() -> list[dict]:
    """Versões registradas não dominadas em F1 macro (maior) x latência (menor)."""
    return pareto_frontier(service.list_models())


@router.post("/api/v5/studio/search", tags=["studio"])
def search(payload: SearchRequest) -> list[dict]:
    """Busca TF-IDF no README e em `docs/` (não é busca semântica nem RAG)."""
    return [
        {"id": doc.id, "metadata": doc.metadata, "text": doc.text[:500]}
        for doc in documentation_index().search(payload.query, payload.k)
    ]


@router.post("/api/v5/studio/evaluate-predictions", tags=["studio"])
def evaluate_predictions(payload: EvaluatePredictionsRequest) -> dict:
    """Estimativa e intervalo de confiança bootstrap (95%) para vetores de rótulos enviados."""
    return bootstrap_metric(payload.y_true, payload.y_pred, payload.metric, payload.iterations, payload.seed)


@router.post("/api/v6/research/experiment-matrix", tags=["research"])
def experiment_matrix(payload: ExperimentRequest) -> dict:
    """Validação cruzada repetida, com o mesmo protocolo, para os modelos pedidos (ou todos)."""
    spec = ExperimentSpec(payload.models, payload.repeats, payload.folds, payload.random_state)
    with invalid_input_as_422():
        check_cv_budget(len(set(payload.models)) or len(available_models()), payload.folds, payload.repeats)
        return ExperimentMatrix(random_state=payload.random_state).run(load_iris_dataframe(), spec)


@router.get("/api/v6/research/model-space", tags=["research"])
def model_space() -> dict:
    return {
        "available": available_models(),
        "registered": sorted({row["name"] for row in service.list_models()}),
    }


@router.post("/api/v6/modeling/robustness", tags=["modeling"])
def robustness(
    model_name: str = Query("xgboost", max_length=128), scale: float = Query(0.01, ge=0, le=2)
) -> dict:
    """Estabilidade das predições do modelo registrado sob ruído gaussiano nas features."""
    record = service.serving_record(model_name)
    artifact = service.load(record)
    with invalid_input_as_422():  # o dataset em que o modelo foi treinado (pode ter saído do catálogo)
        frame, columns, _ = service.reference_frame(model_name)
    features = frame[columns]
    return {"model": record["alias"], "report": perturbation_report(artifact["model"], features, scale=scale)}
