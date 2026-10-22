from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def bootstrap_metric(y_true, y_pred, metric="accuracy", iterations=300, seed=42) -> dict[str, float]:
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    values = []
    fn = accuracy_score if metric == "accuracy" else lambda a, b: f1_score(a, b, average="macro")
    for _ in range(iterations):
        idx = rng.integers(0, len(y), len(y))
        values.append(float(fn(y[idx], p[idx])))
    return {
        "estimate": float(fn(y, p)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "iterations": iterations,
    }
