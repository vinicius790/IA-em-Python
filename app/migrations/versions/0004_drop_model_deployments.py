"""Remove a tabela `model_deployments`, que nunca foi usada.

Os estágios de implantação vivem no registry de modelos (`catalog.json`).

Revision ID: 0004_drop_model_deployments
Revises: 0003_aurora_control_plane
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_drop_model_deployments"
down_revision: str | None = "0003_aurora_control_plane"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    # Bancos criados pelo antigo create_all() não têm os mesmos nomes de índice
    # das migrações; DROP TABLE remove os índices junto, em qualquer dos dois casos.
    op.drop_table("model_deployments")


def downgrade() -> None:
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
