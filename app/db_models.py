from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class IrisRecord(Base):
    __tablename__ = "iris_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    sepal_length: Mapped[float] = mapped_column(Float, nullable=False)
    sepal_width: Mapped[float] = mapped_column(Float, nullable=False)
    petal_length: Mapped[float] = mapped_column(Float, nullable=False)
    petal_width: Mapped[float] = mapped_column(Float, nullable=False)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(128), default="sklearn", nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    dataset_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    lineage_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    model_card_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    __table_args__ = (Index("ix_prediction_logs_canary_arm_id", "canary_id", "canary_arm", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    model_version: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    canary_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    canary_arm: Mapped[str | None] = mapped_column(String(16), nullable=True)
    predicted_class: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    features: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    actual_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    backend: Mapped[str] = mapped_column(String(64), nullable=False)
    params: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tags: Mapped[dict[str, str] | None] = mapped_column(JSON, nullable=True)
    external_run_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class MonitoringReport(Base):
    __tablename__ = "monitoring_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    report_type: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class JobRecord(Base):
    __tablename__ = "job_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    action: Mapped[str] = mapped_column(String(128), index=True)
    actor: Mapped[str] = mapped_column(String(160), default="system", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pipeline_name: Mapped[str] = mapped_column(String(160), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class ApiUser(Base):
    """Usuário da API: um nome, um papel e o hash SHA-256 da sua chave."""

    __tablename__ = "api_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class CanaryDeployment(Base):
    """Deploy canário: divide o tráfego de um modelo entre a versão estável e a candidata."""

    __tablename__ = "canary_deployments"
    # No máximo um canário ativo por modelo, garantido pelo banco (índice único parcial, migração
    # 0006): duas aberturas simultâneas passam pela checagem da aplicação, mas só uma é gravada.
    __table_args__ = (
        Index(
            "uq_canary_deployments_active_model",
            "model_name",
            unique=True,
            sqlite_where=text("status = 'active'"),
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    stable_version: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_version: Mapped[int] = mapped_column(Integer, nullable=False)
    traffic_percent: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    rule: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    decision: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Só as predições com id maior que este contam para as métricas do canário.
    baseline_prediction_id: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_by: Mapped[str] = mapped_column(String(160), default="api", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
