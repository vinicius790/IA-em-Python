from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..ml import build_models
from .task import TaskSpec, infer_task


@dataclass(slots=True)
class AutoMLConfig:
    target: str | None = None
    max_models: int = 8
    folds: int = 3
    random_state: int = 42


class TabularAutoML:
    """Small deterministic AutoML engine suitable for local discovery and CI smoke tests."""

    def __init__(self, config: AutoMLConfig | None = None):
        self.config = config or AutoMLConfig()

    def run(self, frame: pd.DataFrame) -> dict[str, Any]:
        spec = infer_task(frame, self.config.target)
        if spec.task == "classification":
            return self._classification(frame, spec)
        if spec.task == "regression":
            return self._regression(frame, spec)
        return {
            "task": {
                "task": spec.task,
                "target": spec.target,
                "samples": spec.samples,
                "features": spec.features,
                "categorical": spec.categorical,
                "numeric": spec.numeric,
            },
            "results": [],
            "status": "no_target",
        }

    def _classification(self, frame: pd.DataFrame, spec: TaskSpec) -> dict[str, Any]:
        target = spec.target
        assert target is not None
        X = frame.drop(columns=[target])
        X = X.select_dtypes(include=[np.number])
        if X.empty:
            raise ValueError("At least one numeric feature is required by the baseline AutoML engine")
        y = LabelEncoder().fit_transform(frame[target])
        candidates = list(build_models(self.config.random_state).items())[: self.config.max_models]
        folds = min(self.config.folds, *np.bincount(y))
        rows = []
        started = perf_counter()
        cv = StratifiedKFold(n_splits=max(2, folds), shuffle=True, random_state=self.config.random_state)
        for name, model in candidates:
            result = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring={"accuracy": "accuracy", "f1": "f1_macro"},
                n_jobs=1,
                return_train_score=False,
            )
            rows.append(
                {
                    "model": name,
                    "accuracy": round(float(result["test_accuracy"].mean()), 6),
                    "f1_macro": round(float(result["test_f1"].mean()), 6),
                    "fit_time": round(float(result["fit_time"].mean()), 6),
                }
            )
        return {
            "task": "classification",
            "target": target,
            "rows": len(frame),
            "results": sorted(rows, key=lambda r: (-r["f1_macro"], r["fit_time"])),
            "duration_seconds": round(perf_counter() - started, 6),
        }

    def _regression(self, frame: pd.DataFrame, spec: TaskSpec) -> dict[str, Any]:
        target = spec.target
        assert target is not None
        X = frame.drop(columns=[target]).select_dtypes(include=[np.number])
        if X.empty:
            raise ValueError("At least one numeric feature is required by the baseline AutoML engine")
        y = pd.to_numeric(frame[target], errors="raise").to_numpy()
        models = {
            "ridge": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
            "elastic_net": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        ElasticNet(
                            alpha=0.01, l1_ratio=0.2, max_iter=5000, random_state=self.config.random_state
                        ),
                    ),
                ]
            ),
            "random_forest_regressor": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=200, random_state=self.config.random_state, n_jobs=-1
                        ),
                    ),
                ]
            ),
            "extra_trees_regressor": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=200, random_state=self.config.random_state, n_jobs=-1
                        ),
                    ),
                ]
            ),
            "hist_gradient_boosting_regressor": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingRegressor(max_iter=200, random_state=self.config.random_state),
                    ),
                ]
            ),
        }
        rows = []
        cv = KFold(
            n_splits=min(self.config.folds, len(frame)), shuffle=True, random_state=self.config.random_state
        )
        started = perf_counter()
        for name, model in list(models.items())[: self.config.max_models]:
            result = cross_validate(
                model, X, y, cv=cv, scoring={"mae": "neg_mean_absolute_error", "r2": "r2"}, n_jobs=1
            )
            rows.append(
                {
                    "model": name,
                    "mae": round(float(-result["test_mae"].mean()), 6),
                    "r2": round(float(result["test_r2"].mean()), 6),
                    "fit_time": round(float(result["fit_time"].mean()), 6),
                }
            )
        return {
            "task": "regression",
            "target": target,
            "rows": len(frame),
            "results": sorted(rows, key=lambda r: (-r["r2"], r["mae"])),
            "duration_seconds": round(perf_counter() - started, 6),
        }
