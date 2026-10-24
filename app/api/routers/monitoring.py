"""Router do domínio `monitoring`: handlers das rotas versionadas ``/api/vN`` deste domínio."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...analytics.statistics import correlations
from ...api_v2 import _job_dict
from ...config import get_settings
from ...data import load_iris_dataframe
from ...database import database_ok, get_db
from ...db_models import JobRecord, MonitoringReport
from ...integrations import optional_stack_status
from ...ml import available_models
from ...platform.runtime import runtime_snapshot
from ...platform.sbom import minimal_cyclonedx
from ...schemas import DriftResponse, JobRead, MonitoringReportRead, PlatformStatus
from ...services import get_service

router = APIRouter()
settings = get_settings()
service = get_service()


@router.get("/api/v2/platform", response_model=PlatformStatus, tags=["platform"])
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


@router.get("/api/v2/jobs", response_model=list[JobRead], tags=["jobs"])
def jobs(limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)) -> list[dict]:
    records = db.scalars(select(JobRecord).order_by(JobRecord.created_at.desc()).limit(limit)).all()
    return [_job_dict(record) for record in records]


@router.get("/api/v2/jobs/{job_id}", response_model=JobRead, tags=["jobs"])
def job(job_id: str, db: Session = Depends(get_db)) -> dict:
    record = db.get(JobRecord, job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    return _job_dict(record)


@router.get("/api/v2/monitoring/drift/{model_name}", response_model=DriftResponse, tags=["monitoring"])
def drift(model_name: str, db: Session = Depends(get_db)) -> DriftResponse:
    return DriftResponse(**service.drift_report(db, model_name))


@router.get("/api/v2/monitoring/{model_name}", tags=["monitoring"])
def monitoring(model_name: str, db: Session = Depends(get_db)) -> dict:
    return service.monitoring_summary(db, model_name)


@router.post(
    "/api/v2/monitoring/{model_name}/reports",
    response_model=MonitoringReportRead,
    status_code=201,
    tags=["monitoring"],
)
def create_monitoring_report(model_name: str, db: Session = Depends(get_db)):
    return service.snapshot_monitoring(db, model_name)


@router.get(
    "/api/v2/monitoring/{model_name}/reports", response_model=list[MonitoringReportRead], tags=["monitoring"]
)
def monitoring_reports(model_name: str, limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    query = (
        select(MonitoringReport)
        .where(MonitoringReport.model_name == model_name)
        .order_by(MonitoringReport.id.desc())
        .limit(limit)
    )
    return list(db.scalars(query).all())


@router.get("/api/v3/meta", tags=["platform"])
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


@router.get("/api/v4/analytics/correlations", tags=["analytics"])
def analytics_correlations(method: Literal["pearson", "spearman", "kendall"] = "pearson") -> dict:
    return correlations(load_iris_dataframe(), method)


@router.get("/api/v6/platform/sbom", tags=["platform"])
def sbom() -> dict:
    return minimal_cyclonedx()
