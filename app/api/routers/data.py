"""Router do domínio `data`: handlers das rotas versionadas ``/api/vN`` deste domínio."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...api_v1 import _iris_or_404
from ...api_v3 import IRIS_CONTRACT
from ...api_v8 import _csv_text
from ...config import get_settings
from ...data import TARGET_COLUMN, dataset_summary, ensure_local_dataset, load_iris_dataframe
from ...data_catalog.catalog import DatasetCatalog, public_record
from ...data_catalog.csv_profile import profile_csv, training_problems
from ...data_catalog.profiling import profile_dataframe
from ...data_catalog.synthetic import SyntheticSpec, generate
from ...database import get_db
from ...db_models import IrisRecord
from ...errors import invalid_input_as_422
from ...features.engineering import enrich_iris, feature_manifest
from ...features.selection import select_k_best, variance_filter
from ...governance.audit import record_audit
from ...monitoring.anomaly import detect_anomalies
from ...schemas import DataQualityResponse, DatasetSummary, IrisCreate, IrisRead, SyntheticRequest
from ...services import get_service

router = APIRouter()
settings = get_settings()
service = get_service()


@router.get("/api/v1/dataset/summary", response_model=DatasetSummary, tags=["data"])
def summary() -> DatasetSummary:
    return DatasetSummary(**dataset_summary(load_iris_dataframe()))


@router.post("/api/v1/dataset/seed", tags=["data"])
def seed(replace: bool = False, db: Session = Depends(get_db)) -> dict:
    inserted = service.seed_database(db, replace=replace)
    record_audit(db, "dataset_seed", inserted=inserted, replace=replace)
    return {"inserted": inserted, "message": "Dataset seeded" if inserted else "Dataset already populated"}


@router.get("/api/v1/iris", response_model=list[IrisRead], tags=["data"])
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


@router.post("/api/v1/iris", response_model=IrisRead, status_code=status.HTTP_201_CREATED, tags=["data"])
def create_iris(payload: IrisCreate, db: Session = Depends(get_db)):
    row = IrisRecord(**payload.model_dump(), source="api")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/api/v1/iris/{iris_id}", response_model=IrisRead, tags=["data"])
def get_iris(iris_id: int, db: Session = Depends(get_db)):
    return _iris_or_404(db, iris_id)


@router.put("/api/v1/iris/{iris_id}", response_model=IrisRead, tags=["data"])
def update_iris(iris_id: int, payload: IrisCreate, db: Session = Depends(get_db)):
    row = _iris_or_404(db, iris_id)
    for key, value in payload.model_dump().items():
        setattr(row, key, value)
    db.commit()
    db.refresh(row)
    record_audit(db, "iris_update", iris_id=iris_id)
    return row


@router.delete("/api/v1/iris/{iris_id}", response_model=IrisRead, tags=["data"])
def delete_iris(iris_id: int, db: Session = Depends(get_db)):
    row = _iris_or_404(db, iris_id)
    db.delete(row)
    db.commit()
    record_audit(db, "iris_delete", iris_id=iris_id)
    return row


@router.get("/api/v2/data/quality", response_model=DataQualityResponse, tags=["data"])
def data_quality() -> DataQualityResponse:
    return DataQualityResponse(**service.quality_report())


@router.get("/api/v2/features/manifest", tags=["data"])
def features_manifest(engineered: bool = False) -> dict:
    frame = load_iris_dataframe()
    return feature_manifest(enrich_iris(frame) if engineered else frame, target=TARGET_COLUMN)


@router.get("/api/v3/data/profile", tags=["data"])
def data_profile() -> dict:
    return profile_dataframe(load_iris_dataframe())


@router.post("/api/v3/datasets/register", tags=["data"])
def register_dataset(db: Session = Depends(get_db)) -> dict:
    """Valida o CSV do Iris contra o contrato e registra a versão (SHA-256) no catálogo."""
    path = ensure_local_dataset(settings.data_dir / "processed" / "iris.csv")
    with invalid_input_as_422():
        record = DatasetCatalog(settings.data_dir).register("iris", path, IRIS_CONTRACT)
    record_audit(db, "dataset_register", dataset="iris", sha256=record["sha256"])
    return public_record(record)


@router.get("/api/v3/datasets", tags=["data"], deprecated=True)
def list_datasets() -> list[dict]:
    return [public_record(record) for record in DatasetCatalog(settings.data_dir).list()]


@router.get("/api/v4/data/anomalies", tags=["data"])
def anomalies() -> dict:
    return detect_anomalies(load_iris_dataframe())


@router.get("/api/v4/features/selection", tags=["features"])
def selection(k: int = Query(4, ge=1, le=20), method: Literal["anova", "mutual_info"] = "anova") -> dict:
    frame = enrich_iris(load_iris_dataframe())
    return {"variance": variance_filter(frame), "best": select_k_best(frame, TARGET_COLUMN, k, method)}


@router.post("/api/v6/data/synthetic", tags=["data"])
def synthetic(payload: SyntheticRequest) -> dict:
    with invalid_input_as_422():
        frame, metadata = generate(SyntheticSpec(**payload.model_dump()))
    return {"metadata": metadata, "rows": len(frame), "preview": frame.head(8).to_dict(orient="records")}


@router.post("/api/v8/datasets/profile", tags=["datasets"])
async def dataset_profile(request: Request, target: str | None = Query(None, max_length=128)) -> dict:
    """Perfil de um CSV enviado no corpo (`Content-Type: text/csv`), sem gravar nada."""
    text = await _csv_text(request)
    with invalid_input_as_422():
        profile = await run_in_threadpool(profile_csv, text, target)
    return profile | {"training_problems": training_problems(profile)}


@router.post("/api/v8/datasets/upload", status_code=status.HTTP_201_CREATED, tags=["datasets"])
async def dataset_upload(
    request: Request,
    name: str = Query(min_length=1, max_length=64),
    target: str | None = Query(None, max_length=128),
    db: Session = Depends(get_db),
) -> dict:
    """Recebe um CSV no corpo (`Content-Type: text/csv`), faz o perfil e registra no catálogo.

    Sem `target`, o alvo é inferido. Reenviar o mesmo `name` registra uma nova versão.
    """
    text = await _csv_text(request)
    with invalid_input_as_422():
        return public_record(await run_in_threadpool(service.register_dataset, db, name, text, target))


@router.get("/api/v8/datasets", tags=["datasets"])
def datasets() -> list[dict]:
    """Datasets enviados (a versão mais recente de cada nome), sem o perfil completo."""
    latest = {record["name"]: record for record in service.catalog().list() if record.get("target")}
    return [
        {key: value for key, value in public_record(record).items() if key != "profile"}
        for record in latest.values()
    ]


@router.get("/api/v8/datasets/{name}", tags=["datasets"])
def dataset(name: str) -> dict:
    record = service.catalog().get(name)
    if record is None or not record.get("target"):
        raise HTTPException(404, "Dataset not found")
    return public_record(record)
