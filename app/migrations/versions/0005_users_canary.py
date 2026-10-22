"""Usuários da API, deploys canário e versão do modelo em cada predição.

Revision ID: 0005_users_canary
Revises: 0004_drop_model_deployments
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_users_canary"
down_revision: str | None = "0004_drop_model_deployments"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    # Idempotente: um banco adotado do antigo `create_all()` pode já ter sido criado a
    # partir dos modelos ORM atuais, e então estas tabelas e a coluna já existem.
    # No modo offline (`alembic upgrade --sql`) não há banco para inspecionar: gera tudo.
    tables: set[str] = set()
    log_columns: set[str] = set()
    if not op.get_context().as_sql:
        inspector = sa.inspect(op.get_bind())
        tables = set(inspector.get_table_names())
        log_columns = {column["name"] for column in inspector.get_columns("prediction_logs")}
    if "api_users" not in tables:
        _create_api_users()
    if "canary_deployments" not in tables:
        _create_canary_deployments()
    if "model_version" not in log_columns:
        op.add_column("prediction_logs", sa.Column("model_version", sa.Integer(), nullable=True))
        op.create_index("ix_prediction_logs_model_version", "prediction_logs", ["model_version"])


def _create_api_users() -> None:
    op.create_table(
        "api_users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(160), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("name", name="uq_api_users_name"),
        sa.UniqueConstraint("key_hash", name="uq_api_users_key_hash"),
    )


def _create_canary_deployments() -> None:
    op.create_table(
        "canary_deployments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("stable_version", sa.Integer(), nullable=False),
        sa.Column("candidate_version", sa.Integer(), nullable=False),
        sa.Column("traffic_percent", sa.Float(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("rule", sa.JSON(), nullable=False),
        sa.Column("decision", sa.JSON()),
        sa.Column("baseline_prediction_id", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_by", sa.String(160), nullable=False, server_default="api"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("decided_at", sa.DateTime()),
    )
    op.create_index("ix_canary_deployments_model_name", "canary_deployments", ["model_name"])
    op.create_index("ix_canary_deployments_status", "canary_deployments", ["status"])


def downgrade() -> None:
    # O batch recria `prediction_logs` (SQLite antigo não tem DROP COLUMN); DROP TABLE leva os índices.
    # No modo offline o batch do SQLite não tem como refletir a tabela: usa DROP COLUMN (SQLite >= 3.35).
    if op.get_context().as_sql:
        op.drop_index("ix_prediction_logs_model_version", table_name="prediction_logs")
        op.drop_column("prediction_logs", "model_version")
    else:
        with op.batch_alter_table("prediction_logs") as batch:
            batch.drop_index("ix_prediction_logs_model_version")
            batch.drop_column("model_version")
    op.drop_table("canary_deployments")
    op.drop_table("api_users")
