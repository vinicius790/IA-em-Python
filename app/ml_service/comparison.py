"""Comparação de modelos e relatórios (fatia de ``MLService``, issue #9)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..data import FEATURE_COLUMNS, TARGET_COLUMN
from ..governance.audit import record_audit
from ..modeling.comparison import compare_models
from ..platform.hash_utils import canonical_hash
from ..reports.generator import comparison_report_html
from ._base import COMPARISON_RUN_ID, IRIS_DATASET, ServiceBase


class ComparisonMixin(ServiceBase):
    def compare(
        self,
        db: Session,
        models: list[str],
        *,
        dataset: str | None = None,
        folds: int = 5,
        confidence: float = 0.95,
        random_state: int | None = None,
    ) -> dict[str, Any]:
        """Compara modelos por validação cruzada estratificada e grava o relatório JSON e HTML."""
        frame, features, target = self.dataset_frame(dataset)
        report = compare_models(
            frame,
            features or FEATURE_COLUMNS,
            target or TARGET_COLUMN,
            models,
            folds=folds,
            random_state=self.settings.random_state if random_state is None else random_state,
            confidence=confidence,
        )
        created_at = datetime.now(UTC)
        report["dataset"] = dataset or IRIS_DATASET
        report["created_at"] = created_at.isoformat()
        report["run_id"] = f"{created_at:%Y%m%dT%H%M%S}-{canonical_hash(report)[:8]}"
        directory = self._comparison_dir()
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{report['run_id']}.json"
        html_path = comparison_report_html(report, directory / f"{report['run_id']}.html")
        report["reports"] = {"json": str(json_path), "html": str(html_path)}
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        record_audit(
            db,
            "model_comparison",
            run_id=report["run_id"],
            dataset=report["dataset"],
            models=models,
            best_model=report["best_model"],
        )
        return report

    def _comparison_dir(self) -> Path:
        return Path(self.settings.artifact_dir) / "comparisons"

    def list_comparisons(self, limit: int = 50) -> list[dict[str, Any]]:
        files = sorted(self._comparison_dir().glob("*.json"), reverse=True)[:limit]
        rows = []
        for file in files:
            data = json.loads(file.read_text(encoding="utf-8"))
            rows.append(
                {
                    "run_id": data["run_id"],
                    "dataset": data["dataset"],
                    "created_at": data["created_at"],
                    "best_model": data["best_model"],
                    "models": [row["model"] for row in data["results"]],
                }
            )
        return rows

    def comparison_file(self, run_id: str, suffix: str) -> Path | None:
        """Caminho do relatório de uma execução; o `run_id` é validado antes de virar caminho."""
        if not COMPARISON_RUN_ID.fullmatch(run_id):
            return None
        path = self._comparison_dir() / f"{run_id}.{suffix}"
        return path if path.is_file() else None
