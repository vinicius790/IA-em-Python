"""Dataset Iris de referência e qualidade (fatia de ``MLService``, issue #9)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..data import load_iris_dataframe
from ..db_models import IrisRecord
from ..quality import validate_training_frame
from ._base import ServiceBase


class DataMixin(ServiceBase):
    def seed_database(self, db: Session, *, replace: bool = False) -> int:
        if replace:
            db.query(IrisRecord).delete()
            db.commit()
        if db.scalar(select(IrisRecord.id).limit(1)) is not None:
            return 0
        frame = load_iris_dataframe()
        db.add_all([IrisRecord(**row, source="sklearn") for row in frame.to_dict(orient="records")])
        db.commit()
        return len(frame)

    @staticmethod
    def quality_report() -> dict:
        result = validate_training_frame(load_iris_dataframe())
        return {
            "passed": result.passed,
            "score": result.score,
            "checks": result.checks,
            "profile": result.profile,
        }
