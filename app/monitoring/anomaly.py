from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(
    frame: pd.DataFrame, contamination: float | str = "auto", random_state: int = 42
) -> dict[str, Any]:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return {"count": 0, "anomalies": [], "message": "No numeric columns"}
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(numeric)
    scores = model.decision_function(numeric)
    indices = [int(i) for i, v in enumerate(labels) if v == -1]
    return {
        "count": len(indices),
        "rate": len(indices) / len(frame),
        "indices": indices[:100],
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
    }
