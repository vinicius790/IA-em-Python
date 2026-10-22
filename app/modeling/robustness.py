from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    samples: int
    perturbation_scale: float
    changed_predictions: int
    stability: float
    class_counts: dict[str, int]


def perturbation_report(
    model: Any, features: pd.DataFrame, *, scale: float = 0.01, seed: int = 42
) -> dict[str, Any]:
    if features.empty:
        raise ValueError("features must not be empty")
    if scale < 0:
        raise ValueError("scale must be >= 0")
    rng = np.random.default_rng(seed)
    baseline = np.asarray(model.predict(features))
    noise = rng.normal(0.0, scale, size=features.shape)
    perturbed_frame = pd.DataFrame(
        features.to_numpy() + noise, columns=features.columns, index=features.index
    )
    perturbed = np.asarray(model.predict(perturbed_frame))
    changed = int(np.sum(baseline != perturbed))
    unique, counts = np.unique(baseline, return_counts=True)
    report = RobustnessReport(
        len(features),
        scale,
        changed,
        1.0 - changed / len(features),
        {str(k): int(v) for k, v in zip(unique, counts, strict=False)},
    )
    return {
        "samples": report.samples,
        "perturbation_scale": report.perturbation_scale,
        "changed_predictions": report.changed_predictions,
        "stability": round(report.stability, 6),
        "class_counts": report.class_counts,
    }
