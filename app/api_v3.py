"""API v3: metadados da plataforma, perfil e catálogo de datasets."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from .config import get_settings
from .data import FEATURE_COLUMNS, TARGET_COLUMN, ensure_local_dataset, load_iris_dataframe
from .data_catalog.catalog import DatasetCatalog, public_record
from .data_catalog.contracts import ColumnContract, DatasetContract
from .data_catalog.profiling import profile_dataframe
from .database import get_db
from .errors import invalid_input_as_422
from .governance.audit import record_audit
from .ml import available_models
from .platform.runtime import runtime_snapshot

router = APIRouter(prefix="/api/v3")
settings = get_settings()

IRIS_CONTRACT = DatasetContract(
    "iris",
    [
        *(
            ColumnContract(name, "float64", nullable=False, min_value=0, max_value=20)
            for name in FEATURE_COLUMNS
        ),
        ColumnContract(TARGET_COLUMN, nullable=False),
    ],
    min_rows=100,
    target=TARGET_COLUMN,
)


@router.get("/meta", tags=["platform"])
def meta(request: Request) -> dict:
    """Versão, grupos de API realmente montados, modelos disponíveis e ambiente de execução."""
    paths = request.app.openapi()["paths"]
    api_groups = sorted({path.split("/")[2] for path in paths if path.startswith("/api/")})
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "api_groups": api_groups,
        "available_models": available_models(),
        "runtime": runtime_snapshot(),
    }


@router.get("/data/profile", tags=["data"])
def data_profile() -> dict:
    return profile_dataframe(load_iris_dataframe())


@router.post("/datasets/register", tags=["data"])
def register_dataset(db: Session = Depends(get_db)) -> dict:
    """Valida o CSV do Iris contra o contrato e registra a versão (SHA-256) no catálogo."""
    path = ensure_local_dataset(settings.data_dir / "processed" / "iris.csv")
    with invalid_input_as_422():
        record = DatasetCatalog(settings.data_dir).register("iris", path, IRIS_CONTRACT)
    record_audit(db, "dataset_register", dataset="iris", sha256=record["sha256"])
    return public_record(record)


@router.get("/datasets", tags=["data"], deprecated=True)
def list_datasets() -> list[dict]:
    return [public_record(record) for record in DatasetCatalog(settings.data_dir).list()]
