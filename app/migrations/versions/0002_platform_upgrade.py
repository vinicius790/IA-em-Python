"""Add MLOps lineage, experiments, deployments, monitoring and jobs.

Revision ID: 0002_platform_upgrade
Revises: 0001_initial
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_platform_upgrade"
down_revision: str | None = "0001_initial"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "iris_records", sa.Column("source", sa.String(length=128), nullable=False, server_default="sklearn")
    )
    op.create_index("ix_iris_records_source", "iris_records", ["source"])

    for name, typ in [
        ("dataset_hash", sa.String(length=128)),
        ("lineage_hash", sa.String(length=128)),
        ("model_card_path", sa.Text()),
    ]:
        op.add_column("training_runs", sa.Column(name, typ, nullable=True))
    op.create_index("ix_training_runs_dataset_hash", "training_runs", ["dataset_hash"])
    op.create_index("ix_training_runs_lineage_hash", "training_runs", ["lineage_hash"])

    for name, typ in [
        ("latency_ms", sa.Float()),
        ("request_id", sa.String(length=128)),
        ("actual_class", sa.String(length=64)),
    ]:
        op.add_column("prediction_logs", sa.Column(name, typ, nullable=True))
    op.create_index("ix_prediction_logs_request_id", "prediction_logs", ["request_id"])

    op.create_table(
        "experiment_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("backend", sa.String(length=64), nullable=False),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("external_run_id", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_experiment_runs_id", "experiment_runs", ["id"])
    op.create_index("ix_experiment_runs_name", "experiment_runs", ["name"])

    op.create_table(
        "model_deployments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False, server_default="candidate"),
        sa.Column("traffic_percent", sa.Float(), nullable=False, server_default="0"),
        sa.Column("approved_by", sa.String(length=160), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("model_name", name="uq_model_deployment_name"),
    )
    op.create_index("ix_model_deployments_id", "model_deployments", ["id"])
    op.create_index("ix_model_deployments_model_name", "model_deployments", ["model_name"])
    op.create_index("ix_model_deployments_stage", "model_deployments", ["stage"])

    op.create_table(
        "monitoring_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_name", sa.String(length=128), nullable=True),
        sa.Column("report_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_monitoring_reports_id", "monitoring_reports", ["id"])
    op.create_index("ix_monitoring_reports_model_name", "monitoring_reports", ["model_name"])
    op.create_index("ix_monitoring_reports_report_type", "monitoring_reports", ["report_type"])
    op.create_index("ix_monitoring_reports_status", "monitoring_reports", ["status"])

    op.create_table(
        "job_records",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_job_records_kind", "job_records", ["kind"])
    op.create_index("ix_job_records_status", "job_records", ["status"])


def downgrade() -> None:
    op.drop_index("ix_job_records_status", table_name="job_records")
    op.drop_index("ix_job_records_kind", table_name="job_records")
    op.drop_table("job_records")
    op.drop_index("ix_monitoring_reports_status", table_name="monitoring_reports")
    op.drop_index("ix_monitoring_reports_report_type", table_name="monitoring_reports")
    op.drop_index("ix_monitoring_reports_model_name", table_name="monitoring_reports")
    op.drop_index("ix_monitoring_reports_id", table_name="monitoring_reports")
    op.drop_table("monitoring_reports")
    op.drop_index("ix_model_deployments_stage", table_name="model_deployments")
    op.drop_index("ix_model_deployments_model_name", table_name="model_deployments")
    op.drop_index("ix_model_deployments_id", table_name="model_deployments")
    op.drop_table("model_deployments")
    op.drop_index("ix_experiment_runs_name", table_name="experiment_runs")
    op.drop_index("ix_experiment_runs_id", table_name="experiment_runs")
    op.drop_table("experiment_runs")
    op.drop_index("ix_prediction_logs_request_id", table_name="prediction_logs")
    op.drop_column("prediction_logs", "actual_class")
    op.drop_column("prediction_logs", "request_id")
    op.drop_column("prediction_logs", "latency_ms")
    op.drop_index("ix_training_runs_lineage_hash", table_name="training_runs")
    op.drop_index("ix_training_runs_dataset_hash", table_name="training_runs")
    op.drop_column("training_runs", "model_card_path")
    op.drop_column("training_runs", "lineage_hash")
    op.drop_column("training_runs", "dataset_hash")
    op.drop_index("ix_iris_records_source", table_name="iris_records")
    op.drop_column("iris_records", "source")
