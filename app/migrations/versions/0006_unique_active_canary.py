"""No máximo um canário ativo por modelo (índice único parcial).

Revision ID: 0006_unique_active_canary
Revises: 0005_users_canary
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_unique_active_canary"
down_revision: str | None = "0005_users_canary"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

INDEX = "uq_canary_deployments_active_model"
ACTIVE = sa.text("status = 'active'")
# Índice parcial existe no SQLite e no PostgreSQL (os bancos suportados). Em outro dialeto o
# `WHERE` seria ignorado e o índice proibiria o histórico de canários do modelo; lá não é criado.
PARTIAL_INDEX_DIALECTS = {"sqlite", "postgresql"}


def _existing_indexes() -> set[str] | None:
    """Índices da tabela, ou `None` no modo offline (`--sql`), em que não há banco para inspecionar."""
    if op.get_context().as_sql:
        return None
    return {index["name"] for index in sa.inspect(op.get_bind()).get_indexes("canary_deployments")}


def upgrade() -> None:
    if op.get_context().dialect.name not in PARTIAL_INDEX_DIALECTS:
        return
    existing = _existing_indexes()
    if existing is not None and INDEX in existing:
        return
    # Um banco em que a corrida já aconteceu tem dois ativos para o mesmo modelo: fica o mais novo.
    op.execute(
        "UPDATE canary_deployments SET status = 'rolled_back', decided_at = CURRENT_TIMESTAMP, "
        """decision = '{"decision": "rollback", "reason": "duplicate active canary closed by migration 0006"}' """
        "WHERE status = 'active' AND id NOT IN "
        "(SELECT MAX(id) FROM canary_deployments WHERE status = 'active' GROUP BY model_name)"
    )
    op.create_index(
        INDEX,
        "canary_deployments",
        ["model_name"],
        unique=True,
        sqlite_where=ACTIVE,
        postgresql_where=ACTIVE,
    )


def downgrade() -> None:
    if op.get_context().dialect.name not in PARTIAL_INDEX_DIALECTS:
        return
    existing = _existing_indexes()
    if existing is None or INDEX in existing:
        op.drop_index(INDEX, table_name="canary_deployments")
