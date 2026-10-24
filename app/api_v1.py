"""API v1: saúde, dados, treino síncrono, inferência e consulta ao registry."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .data import dataset_summary, load_iris_dataframe
from .database import database_ok, get_db
from .db_models import IrisRecord, TrainingRun
from .governance.audit import record_audit
from .ml import available_models
from .schemas import (
    BatchPredictionRequest,
    DatasetSummary,
    HealthResponse,
    IrisCreate,
    IrisRead,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
    TrainingRequest,
    TrainingResponse,
    TrainingRunRead,
)
from .services import get_service

# Rotas públicas (sem API key): sondas de saúde, métricas e o atalho para o console.
public_router = APIRouter(prefix="/api/v1")
router = APIRouter(prefix="/api/v1")
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


@router.get("/dataset/summary", response_model=DatasetSummary, tags=["data"])
def summary() -> DatasetSummary:
    return DatasetSummary(**dataset_summary(load_iris_dataframe()))


@router.post("/dataset/seed", tags=["data"])
def seed(replace: bool = False, db: Session = Depends(get_db)) -> dict:
    inserted = service.seed_database(db, replace=replace)
    record_audit(db, "dataset_seed", inserted=inserted, replace=replace)
    return {"inserted": inserted, "message": "Dataset seeded" if inserted else "Dataset already populated"}


@router.get("/iris", response_model=list[IrisRead], tags=["data"])
def list_iris(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    class_name: str | None = None,
    db: Session = Depends(get_db),
):
    stmt = select(IrisRecord).order_by(IrisRecord.id).offset(skip).limit(limit)
    if class_name:
        stmt = stmt.where(IrisRecord.class_name == class_name)
    return list(db.scalars(stmt).all())


@router.post("/iris", response_model=IrisRead, status_code=status.HTTP_201_CREATED, tags=["data"])
def create_iris(payload: IrisCreate, db: Session = Depends(get_db)):
    row = IrisRecord(**payload.model_dump(), source="api")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _iris_or_404(db: Session, iris_id: int) -> IrisRecord:
    row = db.get(IrisRecord, iris_id)
    if row is None:
        raise HTTPException(404, "Iris record not found")
    return row


@router.get("/iris/{iris_id}", response_model=IrisRead, tags=["data"])
def get_iris(iris_id: int, db: Session = Depends(get_db)):
    return _iris_or_404(db, iris_id)


@router.put("/iris/{iris_id}", response_model=IrisRead, tags=["data"])
def update_iris(iris_id: int, payload: IrisCreate, db: Session = Depends(get_db)):
    row = _iris_or_404(db, iris_id)
    for key, value in payload.model_dump().items():
        setattr(row, key, value)
    db.commit()
    db.refresh(row)
    record_audit(db, "iris_update", iris_id=iris_id)
    return row


@router.delete("/iris/{iris_id}", response_model=IrisRead, tags=["data"])
def delete_iris(iris_id: int, db: Session = Depends(get_db)):
    row = _iris_or_404(db, iris_id)
    db.delete(row)
    db.commit()
    record_audit(db, "iris_delete", iris_id=iris_id)
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


@router.post("/training/train", response_model=TrainingResponse, tags=["machine-learning"])
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


@router.get("/training/runs", response_model=list[TrainingRunRead], tags=["machine-learning"])
def training_runs(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    return list(db.scalars(select(TrainingRun).order_by(TrainingRun.id.desc()).limit(limit)).all())


@router.post("/predict", response_model=PredictionResponse, tags=["inference"], deprecated=True)
def predict(payload: PredictionRequest, request: Request, db: Session = Depends(get_db)):
    return service.predict(
        db,
        payload.model_name or settings.default_model,
        payload.as_list(),
        request_id=request.state.request_id,
    )


@router.post("/predict/batch", response_model=list[PredictionResponse], tags=["inference"])
def predict_batch(payload: BatchPredictionRequest, request: Request, db: Session = Depends(get_db)):
    if len(payload.items) > settings.max_batch_size:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"Batch exceeds max size of {settings.max_batch_size}"
        )
    items = [(item.model_name or settings.default_model, item.as_list()) for item in payload.items]
    return service.predict_batch(db, items, request_id=request.state.request_id)


@router.get("/models", tags=["machine-learning"])
def models() -> list[dict]:
    return service.list_models()


@router.get("/models/{model_name}", tags=["machine-learning"])
def model(model_name: str) -> dict:
    record = service.latest_model(model_name)
    if not record:
        raise HTTPException(404, "Model not found")
    return record
