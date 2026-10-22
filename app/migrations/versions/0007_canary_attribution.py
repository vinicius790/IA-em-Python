"""Atribui cada observação de predição ao canário e braço que a geraram.

Revision ID: 0007_canary_attribution
Revises: 0006_unique_active_canary
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_canary_attribution"
down_revision: str | None = "0006_unique_active_canary"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

TABLE = "prediction_logs"
INDEX = "ix_prediction_logs_canary_arm_id"


def upgrade() -> None:
    # O banco legado pode ter sido criado por `create_all()` com os campos do modelo atual.
    # No modo offline não há como inspecionar; o caminho segue a cadeia normal de migrações.
    if op.get_context().as_sql:
        op.add_column(TABLE, sa.Column("canary_id", sa.Integer(), nullable=True))
        op.add_column(TABLE, sa.Column("canary_arm", sa.String(length=16), nullable=True))
        op.create_index(INDEX, TABLE, ["canary_id", "canary_arm", "id"])
        return

    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    if "canary_id" not in columns:
        op.add_column(TABLE, sa.Column("canary_id", sa.Integer(), nullable=True))
    if "canary_arm" not in columns:
        op.add_column(TABLE, sa.Column("canary_arm", sa.String(length=16), nullable=True))
    indexes = {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(TABLE)}
    if INDEX not in indexes:
        op.create_index(INDEX, TABLE, ["canary_id", "canary_arm", "id"])


def downgrade() -> None:
    if op.get_context().as_sql:
        op.drop_index(INDEX, table_name=TABLE)
        op.drop_column(TABLE, "canary_arm")
        op.drop_column(TABLE, "canary_id")
        return

    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns(TABLE)}
    indexes = {index["name"] for index in inspector.get_indexes(TABLE)}
    if INDEX not in indexes and not {"canary_arm", "canary_id"}.intersection(columns):
        return
    with op.batch_alter_table(TABLE) as batch:
        if INDEX in indexes:
            batch.drop_index(INDEX)
        if "canary_arm" in columns:
            batch.drop_column("canary_arm")
        if "canary_id" in columns:
            batch.drop_column("canary_id")
