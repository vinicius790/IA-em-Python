from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder

from ..data import FEATURE_COLUMNS, TARGET_COLUMN
from ..ml import build_models


@dataclass(slots=True)
class ExperimentSpec:
    models: list[str] = field(default_factory=list)
    repeats: int = 2
    folds: int = 5
    random_state: int = 42

    def normalized(self, available: list[str]) -> ExperimentSpec:
        selected = self.models or available
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(f"Unknown models: {', '.join(unknown)}")
        if self.repeats < 1 or self.folds < 2:
            raise ValueError("repeats must be >=1 and folds must be >=2")
        return ExperimentSpec(selected, self.repeats, self.folds, self.random_state)


class ExperimentMatrix:
    """Runs comparable repeated cross-validation experiments with the same protocol."""

    def __init__(self, *, random_state: int = 42):
        self.random_state = random_state

    def run(self, frame: pd.DataFrame, spec: ExperimentSpec | None = None) -> dict[str, Any]:
        models = build_models(self.random_state)
        spec = (spec or ExperimentSpec(random_state=self.random_state)).normalized(sorted(models))
        X = frame[FEATURE_COLUMNS]
        y = LabelEncoder().fit_transform(frame[TARGET_COLUMN])
        splitter = RepeatedStratifiedKFold(
            n_splits=spec.folds,
            n_repeats=spec.repeats,
            random_state=spec.random_state,
        )
        rows: list[dict[str, Any]] = []
        started = perf_counter()
        for name in spec.models:
            model_started = perf_counter()
            result = cross_validate(
                models[name],
                X,
                y,
                cv=splitter,
                scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
                n_jobs=1,
                return_train_score=False,
            )
            rows.append(
                {
                    "model": name,
                    "accuracy_mean": round(float(np.mean(result["test_accuracy"])), 6),
                    "accuracy_std": round(float(np.std(result["test_accuracy"])), 6),
                    "f1_macro_mean": round(float(np.mean(result["test_f1_macro"])), 6),
                    "f1_macro_std": round(float(np.std(result["test_f1_macro"])), 6),
                    "fit_time_mean": round(float(np.mean(result["fit_time"])), 6),
                    "score_time_mean": round(float(np.mean(result["score_time"])), 6),
                    "duration_seconds": round(perf_counter() - model_started, 6),
                    "folds": spec.folds * spec.repeats,
                }
            )
        return {
            "protocol": {
                "folds": spec.folds,
                "repeats": spec.repeats,
                "random_state": spec.random_state,
                "dataset_rows": len(frame),
            },
            "results": rows,
            "duration_seconds": round(perf_counter() - started, 6),
        }

    @staticmethod
    def metric_delta(y_true: list[Any], first: list[Any], second: list[Any]) -> dict[str, float]:
        """Compare two fixed prediction vectors using paired observations."""
        if not (len(y_true) == len(first) == len(second)):
            raise ValueError("Prediction vectors must have equal length")
        return {
            "accuracy_first": float(accuracy_score(y_true, first)),
            "accuracy_second": float(accuracy_score(y_true, second)),
            "f1_first": float(f1_score(y_true, first, average="macro")),
            "f1_second": float(f1_score(y_true, second, average="macro")),
            "accuracy_delta": float(accuracy_score(y_true, first) - accuracy_score(y_true, second)),
            "f1_delta": float(
                f1_score(y_true, first, average="macro") - f1_score(y_true, second, average="macro")
            ),
        }
