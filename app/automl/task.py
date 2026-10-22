from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task: str
    target: str | None
    samples: int
    features: int
    categorical: int
    numeric: int


def infer_task(frame: pd.DataFrame, target: str | None = None) -> TaskSpec:
    if frame.empty:
        raise ValueError("Dataset must not be empty")
    target = target or (str(frame.columns[-1]) if frame.shape[1] > 1 else None)
    if target is None or target not in frame.columns:
        return TaskSpec(
            "unsupervised",
            None,
            len(frame),
            frame.shape[1],
            int(frame.dtypes.eq("object").sum()),
            int(frame.shape[1]),
        )
    y = frame[target]
    unique = int(y.nunique(dropna=True))
    task = (
        "classification"
        if y.dtype == "object" or y.dtype.name == "category" or unique <= min(20, max(2, len(frame) // 10))
        else "regression"
    )
    features = frame.drop(columns=[target])
    categorical = sum(1 for dtype in features.dtypes if dtype == "object" or str(dtype) == "category")
    return TaskSpec(task, target, len(frame), features.shape[1], categorical, features.shape[1] - categorical)
