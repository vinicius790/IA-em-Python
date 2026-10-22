from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..db_models import PipelineRun
from ..deployment.gates import evaluate_release_gate
from ..governance.audit import record_audit
from ..reports.generator import training_report_html
from ..services import MLService


class TrainingPipeline:
    """Pipeline ponta a ponta: qualidade -> treino -> release gate -> relatório HTML.

    Cada execução fica gravada em `pipeline_runs`.
    """

    name = "iris-training"

    def __init__(self, service: MLService):
        self.service = service

    def run(
        self,
        db: Session,
        model_name: str = "xgboost",
        test_size: float = 0.2,
        cv_folds: int = 5,
        tune: bool = False,
        trials: int = 12,
    ) -> dict[str, Any]:
        quality = self.service.quality_report()
        if not quality["passed"]:
            result: dict[str, Any] = {"status": "blocked", "quality_checks": quality["checks"]}
        else:
            settings = self.service.settings
            run = self.service.train(db, model_name, test_size, cv_folds, tune, trials)
            metrics = run.metrics or {}
            if run.status != "completed":
                result = {"status": "failed", "run_id": run.id, "error": run.error_message}
            else:
                gate = evaluate_release_gate(
                    metrics,
                    min_accuracy=settings.gate_min_accuracy,
                    min_f1=settings.gate_min_f1,
                    max_latency_ms=settings.gate_max_latency_ms,
                )
                report = training_report_html(
                    {"model_name": model_name, "metrics": metrics, "run_id": run.id, "release_gate": gate},
                    Path(settings.artifact_dir) / "reports" / f"{model_name}-{run.id}.html",
                )
                result = {
                    "status": "completed",
                    "run_id": run.id,
                    "metrics": metrics,
                    "quality_score": quality["score"],
                    "release_gate": gate,
                    "report_path": str(report),
                    "artifact_path": run.artifact_path,
                    "model_card_path": run.model_card_path,
                }
        summary = {key: value for key, value in result.items() if key != "metrics"}
        row = PipelineRun(
            pipeline_name=self.name, status=result["status"], model_name=model_name, payload=summary
        )
        db.add(row)
        db.commit()
        record_audit(
            db, "training_pipeline", model=model_name, status=result["status"], pipeline_run_id=row.id
        )
        return result | {"pipeline_run_id": row.id}
