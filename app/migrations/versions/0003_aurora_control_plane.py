"""Tabelas de auditoria e de execuções de pipeline.

Revision ID: 0003_aurora_control_plane
Revises: 0002_platform_upgrade
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_aurora_control_plane"
down_revision: str | None = "0002_platform_upgrade"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("action", sa.String(128), nullable=False),
        sa.Column("actor", sa.String(160), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_audit_events_action", "audit_events", ["action"])
    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("pipeline_name", sa.String(160), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("model_name", sa.String(128)),
        sa.Column("payload", sa.JSON()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_pipeline_runs_pipeline_name", "pipeline_runs", ["pipeline_name"])
    op.create_index("ix_pipeline_runs_status", "pipeline_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_pipeline_runs_status", table_name="pipeline_runs")
    op.drop_index("ix_pipeline_runs_pipeline_name", table_name="pipeline_runs")
    op.drop_table("pipeline_runs")
    op.drop_index("ix_audit_events_action", table_name="audit_events")
    op.drop_table("audit_events")
