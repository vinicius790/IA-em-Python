"""Inferência, explicação e feedback (fatia de ``MLService``, issue #9)."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from ..data import FEATURE_COLUMNS, load_iris_dataframe
from ..db_models import PredictionLog
from ..deployment.canary import validate_canary_attribution
from ..explainability import shapley_explanation
from ..ml import predict_artifact
from ..observability import PREDICTION_COUNT, PREDICTION_LATENCY
from ..platform.cache import prediction_cache
from ..tracing import set_span_attributes, traced
from ._base import ServiceBase


class InferenceMixin(ServiceBase):
    @traced("iris.predict")
    def predict_batch(
        self,
        db: Session,
        items: list[tuple[str, list[float]]],
        *,
        request_id: str | None = None,
        records: dict[str, dict[str, Any]] | None = None,
        canary_id: int | None = None,
        canary_arm: str | None = None,
    ) -> list[dict]:
        """Inferência em lote: uma carga de artefato e uma predição vetorizada por modelo.

        `records` fixa a versão servida de um modelo (é como o canário escolhe o braço);
        sem ele vale `serving_record`.
        """
        canary_id, canary_arm = validate_canary_attribution(canary_id, canary_arm)
        positions: dict[str, list[int]] = {}
        for position, (model_name, _) in enumerate(items):
            positions.setdefault(model_name, []).append(position)

        logs: list[PredictionLog | None] = [None] * len(items)
        results: list[dict | None] = [None] * len(items)
        for model_name, indexes in positions.items():
            record = (records or {}).get(model_name) or self.serving_record(model_name)
            artifact = self.load(record)
            set_span_attributes(model_name=model_name, model_version=record["version"], batch_size=len(items))
            for i in indexes:
                self._check_width(artifact, record, items[i][1])
            labels = [str(label) for label in artifact["label_encoder"].classes_]
            cached_data: dict[int, tuple[str, float, dict[str, float], float]] = {}
            uncached_indexes: list[int] = []

            for i in indexes:
                cached = prediction_cache.get(model_name, record["version"], items[i][1])
                if cached is not None:
                    cached_data[i] = (cached[0], cached[1], cached[2], 0.05)
                else:
                    uncached_indexes.append(i)

            if uncached_indexes:
                started = perf_counter()
                classes, probabilities = self.predict_rows(
                    record, artifact, [items[i][1] for i in uncached_indexes]
                )
                measured_latency = round((perf_counter() - started) * 1000 / len(uncached_indexes), 4)
                for offset, position in enumerate(uncached_indexes):
                    probability_map = {
                        label: round(float(p), 8)
                        for label, p in zip(labels, probabilities[offset], strict=True)
                    }
                    confidence = max(probability_map.values())
                    predicted_class = classes[offset]
                    pred_entry = (predicted_class, confidence, probability_map)
                    prediction_cache.set(model_name, record["version"], items[position][1], pred_entry)
                    cached_data[position] = (predicted_class, confidence, probability_map, measured_latency)

            for position in indexes:
                predicted_class, confidence, probability_map, latency_ms = cached_data[position]
                logs[position] = PredictionLog(
                    model_name=model_name,
                    model_version=record["version"],
                    canary_id=canary_id,
                    canary_arm=canary_arm,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    features=items[position][1],
                    latency_ms=latency_ms,
                    request_id=request_id,
                )
                results[position] = {
                    "model_name": model_name,
                    "model_version": record["version"],
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": probability_map,
                    "request_id": request_id,
                    "latency_ms": latency_ms,
                }
                PREDICTION_COUNT.labels(model_name, predicted_class).inc()
                PREDICTION_LATENCY.labels(model_name).observe(latency_ms / 1000)
        db.add_all(logs)
        db.commit()
        for log, result in zip(logs, results, strict=True):
            result["prediction_id"] = log.id
        return results

    def predict(
        self, db: Session, model_name: str, features: list[float], *, request_id: str | None = None
    ) -> dict:
        return self.predict_batch(db, [(model_name, features)], request_id=request_id)[0]

    def score_dataframe(self, model_name: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Pontua um DataFrame fora da API (CLI `batch-score`), sem gravar logs de predição."""
        record = self.serving_record(model_name)
        artifact = self.load(record)
        classes, probabilities = self.predict_rows(
            record, artifact, frame[artifact["features"]].astype(float).values.tolist()
        )
        return frame.assign(prediction=classes, confidence=probabilities.max(axis=1))

    def record_feedback(self, db: Session, prediction_id: int, actual_class: str) -> PredictionLog:
        """Registra o rótulo real de uma predição; alimenta a acurácia observada."""
        log = db.get(PredictionLog, prediction_id)
        if log is None:
            raise KeyError(f"Prediction {prediction_id} not found")
        log.actual_class = actual_class
        db.commit()
        return log

    def explain(self, model_name: str, features: list[float]) -> dict:
        """Explica a predição desta instância (Shapley exato) e traz a importância global do treino."""
        record = self.serving_record(model_name)
        artifact = self.load(record)
        self._check_width(artifact, record, features)
        background = artifact.get("background")
        if background is None:  # artefatos anteriores à 8.0 não guardam a amostra de fundo
            background = load_iris_dataframe()[FEATURE_COLUMNS].sample(
                100, random_state=self.settings.random_state
            )
        _, probabilities = predict_artifact(artifact, [features])
        class_index = int(probabilities[0].argmax())
        explanation = shapley_explanation(artifact["model"], background, features, class_index)
        explanation["predicted_class"] = str(artifact["label_encoder"].classes_[class_index])
        explanation["global_importance"] = (record.get("metrics") or {}).get("feature_importance", [])
        return {"model_name": model_name, "model_version": record["version"], "explanation": explanation}
