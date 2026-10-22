"""Comparação de modelos por validação cruzada estratificada, com intervalo de confiança."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder

from ..ml import build_models
from .intervals import confidence_interval

METRICS = ("f1_macro", "accuracy")


def compare_models(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    models: list[str],
    *,
    folds: int = 5,
    random_state: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Avalia todos os modelos nas mesmas dobras e compara cada um com o melhor.

    Como as dobras são idênticas, a diferença para o melhor é pareada: o intervalo de
    confiança é calculado sobre as diferenças dobra a dobra. `differs_from_best` é
    verdadeiro quando esse intervalo não contém zero.
    """
    catalog = build_models(random_state)
    unknown = sorted(set(models) - set(catalog))
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(unknown)}. Available: {', '.join(sorted(catalog))}")
    if len(set(models)) < 2:
        raise ValueError("At least two distinct models are required for a comparison")
    X = frame[features]
    y = LabelEncoder().fit_transform(frame[target])
    smallest_class = int(np.bincount(y).min())
    if smallest_class < folds:
        raise ValueError(f"folds={folds} exceeds the smallest class size ({smallest_class})")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    started = perf_counter()
    scores: dict[str, dict[str, list[float]]] = {}
    rows: list[dict[str, Any]] = []
    for name in dict.fromkeys(models):
        result = cross_validate(
            catalog[name], X, y, cv=splitter, scoring={metric: metric for metric in METRICS}, n_jobs=1
        )
        scores[name] = {metric: [float(v) for v in result[f"test_{metric}"]] for metric in METRICS}
        rows.append(
            {
                "model": name,
                **{metric: confidence_interval(scores[name][metric], confidence) for metric in METRICS},
                "fold_scores": {metric: [round(v, 6) for v in scores[name][metric]] for metric in METRICS},
                "fit_time_mean": round(float(np.mean(result["fit_time"])), 6),
            }
        )
    rows.sort(key=lambda row: (-row["f1_macro"]["mean"], row["fit_time_mean"]))
    best = rows[0]["model"]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        differences = [
            b - s for b, s in zip(scores[best]["f1_macro"], scores[row["model"]]["f1_macro"], strict=True)
        ]
        interval = confidence_interval(differences, confidence)
        row["f1_macro_delta_to_best"] = interval
        row["differs_from_best"] = bool(interval["lower"] > 0 or interval["upper"] < 0)
    return {
        "protocol": {
            "folds": folds,
            "random_state": random_state,
            "confidence": confidence,
            "rows": len(frame),
            "features": list(features),
            "target": target,
            "classes": int(len(np.unique(y))),
            "interval": "student-t over fold scores",
        },
        "best_model": best,
        "results": rows,
        "duration_seconds": round(perf_counter() - started, 6),
    }
