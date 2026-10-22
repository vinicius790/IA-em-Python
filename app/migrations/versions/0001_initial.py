"""initial schema

Revision ID: 0001_initial
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "iris_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("sepal_length", sa.Float(), nullable=False),
        sa.Column("sepal_width", sa.Float(), nullable=False),
        sa.Column("petal_length", sa.Float(), nullable=False),
        sa.Column("petal_width", sa.Float(), nullable=False),
        sa.Column("class_name", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_iris_records_id", "iris_records", ["id"])
    op.create_index("ix_iris_records_class_name", "iris_records", ["class_name"])

    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("parameters", sa.JSON(), nullable=True),
        sa.Column("artifact_path", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_training_runs_id", "training_runs", ["id"])
    op.create_index("ix_training_runs_model_name", "training_runs", ["model_name"])
    op.create_index("ix_training_runs_status", "training_runs", ["status"])

    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("predicted_class", sa.String(length=64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_prediction_logs_id", "prediction_logs", ["id"])
    op.create_index("ix_prediction_logs_model_name", "prediction_logs", ["model_name"])
    op.create_index("ix_prediction_logs_predicted_class", "prediction_logs", ["predicted_class"])


def downgrade() -> None:
    op.drop_index("ix_prediction_logs_predicted_class", table_name="prediction_logs")
    op.drop_index("ix_prediction_logs_model_name", table_name="prediction_logs")
    op.drop_index("ix_prediction_logs_id", table_name="prediction_logs")
    op.drop_table("prediction_logs")
    op.drop_index("ix_training_runs_status", table_name="training_runs")
    op.drop_index("ix_training_runs_model_name", table_name="training_runs")
    op.drop_index("ix_training_runs_id", table_name="training_runs")
    op.drop_table("training_runs")
    op.drop_index("ix_iris_records_class_name", table_name="iris_records")
    op.drop_index("ix_iris_records_id", table_name="iris_records")
    op.drop_table("iris_records")
