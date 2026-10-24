"""Monitoramento, drift e saúde (fatia de ``MLService``, issue #9)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..data import FEATURE_COLUMNS
from ..db_models import MonitoringReport, PredictionLog
from ..errors import ModelNotFoundError
from ..monitoring.drift import drift_report
from ..monitoring.health import score_health
from ..monitoring.sla import SLO, SLOEvaluator
from ..observability import DRIFT_COUNT, DRIFT_PSI, HEALTH_SCORE
from ..quality import validate_training_frame
from ..tracing import set_span_attributes, traced
from ._base import MIN_DRIFT_SAMPLES, MONITORING_WINDOW, ServiceBase


class MonitoringMixin(ServiceBase):
    def _recent_predictions(self, db: Session, model_name: str) -> list[PredictionLog]:
        query = (
            select(PredictionLog)
            .where(PredictionLog.model_name == model_name)
            .order_by(PredictionLog.id.desc())
            .limit(MONITORING_WINDOW)
        )
        return list(db.scalars(query).all())

    def reference_frame(self, model_name: str) -> tuple[pd.DataFrame, list[str], str]:
        """Dataset (na versão exata) em que o modelo servido foi treinado, com features e alvo."""
        record = self.serving_record(model_name)
        artifact = self.load(record)
        dataset = (artifact.get("metadata") or {}).get("dataset")
        frame, _, _ = self.dataset_frame(dataset, (record.get("metadata") or {}).get("dataset_sha256"))
        return frame, list(artifact["features"]), artifact["target"]

    @traced("iris.drift")
    def _drift(self, model_name: str, rows: list[PredictionLog]) -> dict[str, Any]:
        if len(rows) < MIN_DRIFT_SAMPLES:
            return {"status": "insufficient_data", "samples": len(rows), "required": MIN_DRIFT_SAMPLES}
        try:
            reference, features, _ = self.reference_frame(model_name)
        except ValueError as exc:  # o dataset enviado foi removido do catálogo
            return {"status": "reference_unavailable", "samples": len(rows), "detail": str(exc)}
        current = pd.DataFrame(
            [row.features for row in rows if len(row.features) == len(features)], columns=features
        )
        report = drift_report(
            reference,
            current,
            features,
            warning=self.settings.drift_psi_warning,
            critical=self.settings.drift_psi_threshold,
        )
        report["samples"] = len(rows)
        set_span_attributes(model_name=model_name, drift_status=report.get("status"), samples=len(rows))
        return report

    def drift_report(self, db: Session, model_name: str) -> dict:
        """PSI das features das predições recentes contra o dataset de referência."""
        report = self._drift(model_name, self._recent_predictions(db, model_name))
        return {"model_name": model_name, "status": report["status"], "report": report}

    def monitoring_summary(self, db: Session, model_name: str) -> dict[str, Any]:
        """Resumo medido das últimas predições: confiança, latência, acurácia rotulada, drift, SLOs e saúde."""
        rows = self._recent_predictions(db, model_name)
        if not rows:
            return {"model_name": model_name, "status": "no_data", "predictions": 0}
        latencies = [row.latency_ms for row in rows if row.latency_ms is not None]
        labeled = [row for row in rows if row.actual_class]
        observed: dict[str, float] = {"mean_confidence": float(np.mean([row.confidence for row in rows]))}
        if latencies:
            observed["p95_latency_ms"] = float(np.percentile(latencies, 95))
        if labeled:
            observed["accuracy_when_labeled"] = sum(
                r.predicted_class == r.actual_class for r in labeled
            ) / len(labeled)
        slo = SLOEvaluator(
            [
                SLO("p95_latency_ms", self.settings.slo_p95_latency_ms, "lte"),
                SLO("mean_confidence", self.settings.slo_min_mean_confidence, "gte"),
                SLO("accuracy_when_labeled", self.settings.gate_min_accuracy, "gte"),
            ]
        ).evaluate(observed)
        drift = self._drift(model_name, rows)
        health = score_health(
            quality_passed=self._reference_quality(model_name),
            drift_status=drift["status"],
            failed_slos=[check["name"] for check in slo["checks"] if check.get("passed") is False],
        )
        return {
            "model_name": model_name,
            "status": health["status"],
            "predictions": len(rows),
            "window": MONITORING_WINDOW,
            "labeled": len(labeled),
            "observed": {key: round(value, 6) for key, value in observed.items()},
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 6) if latencies else None,
            "class_distribution": dict(Counter(row.predicted_class for row in rows)),
            "drift": drift,
            "slo": slo,
            "health": health,
        }

    def _reference_quality(self, model_name: str) -> bool:
        try:
            frame, features, target = self.reference_frame(model_name)
        except (ValueError, ModelNotFoundError):
            return False
        iris = features == FEATURE_COLUMNS
        return validate_training_frame(frame, None if iris else features, None if iris else target).passed

    def snapshot_monitoring(self, db: Session, model_name: str) -> MonitoringReport:
        """Persiste o resumo atual em `monitoring_reports` e publica os gauges do Prometheus."""
        summary = self.monitoring_summary(db, model_name)
        report = MonitoringReport(
            model_name=model_name, report_type="monitoring", status=summary["status"], payload=summary
        )
        db.add(report)
        db.commit()
        if summary["predictions"]:
            DRIFT_COUNT.labels(summary["drift"]["status"]).inc()
            HEALTH_SCORE.labels(model_name).set(summary["health"]["score"])
            DRIFT_PSI.labels(model_name).set(summary["drift"].get("max_psi", 0.0))
        return report
