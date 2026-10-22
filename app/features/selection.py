from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif


def variance_filter(frame: pd.DataFrame, threshold: float = 0.0) -> dict[str, Any]:
    numeric = frame.select_dtypes(include="number").copy()
    selector = VarianceThreshold(threshold).fit(numeric)
    return {
        "selected": [c for c, keep in zip(numeric.columns, selector.get_support(), strict=False) if keep],
        "dropped": [c for c, keep in zip(numeric.columns, selector.get_support(), strict=False) if not keep],
    }


class VarianceThresholdSelector:
    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def fit_transform(self, frame: pd.DataFrame) -> dict[str, Any]:
        return variance_filter(frame, self.threshold)


class CorrelationFilter:
    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold

    def fit_transform(self, frame: pd.DataFrame) -> dict[str, Any]:
        numeric = frame.select_dtypes(include="number")
        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return {"dropped": to_drop, "selected": [c for c in numeric.columns if c not in to_drop]}


def select_k_best(frame: pd.DataFrame, target: str, k: int = 4, method: str = "anova") -> dict[str, Any]:
    X = frame.drop(columns=[target])
    X = X.select_dtypes(include="number")
    y = frame[target].astype("category").cat.codes
    score_fn = mutual_info_classif if method == "mutual_info" else f_classif
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=score_fn, k=k).fit(X, y)
    scores = {c: float(s) for c, s in zip(X.columns, selector.scores_, strict=False)}
    return {
        "selected": [c for c, keep in zip(X.columns, selector.get_support(), strict=False) if keep],
        "scores": scores,
        "method": method,
    }
