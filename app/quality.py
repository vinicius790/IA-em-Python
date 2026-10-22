from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data import FEATURE_COLUMNS, TARGET_COLUMN
from .data_catalog.csv_profile import MAX_DUPLICATE_RATE, MIN_ROWS
from .data_catalog.profiling import profile_dataframe


@dataclass(slots=True)
class QualityResult:
    passed: bool
    score: float
    checks: list[dict[str, Any]]
    profile: dict[str, Any]


def validate_training_frame(
    frame: pd.DataFrame, features: list[str] | None = None, target: str | None = None
) -> QualityResult:
    """Gates de qualidade que o dataset precisa passar antes de qualquer treino.

    Sem `features`/`target` valida o Iris, incluindo a regra de medidas positivas, que só
    faz sentido para ele; datasets enviados passam pelos demais gates.
    """
    iris = features is None
    features = list(features or FEATURE_COLUMNS)
    target = target or TARGET_COLUMN
    required = [*features, target]
    missing_columns = [column for column in required if column not in frame.columns]
    checks: list[dict[str, Any]] = [
        {"name": "required_columns", "passed": not missing_columns, "observed": missing_columns},
        {"name": "min_rows", "passed": len(frame) >= MIN_ROWS, "observed": len(frame), "threshold": MIN_ROWS},
    ]
    if not missing_columns:
        data = frame[required]
        nulls = int(data.isna().sum().sum())
        duplicate_rate = float(data.duplicated().mean()) if len(data) else 0.0
        classes = int(data[target].nunique(dropna=True))
        numeric = all(pd.api.types.is_numeric_dtype(data[column]) for column in features)
        checks += [
            {"name": "missing_values", "passed": nulls == 0, "observed": nulls, "threshold": 0},
            {
                "name": "duplicate_rows",
                "passed": duplicate_rate <= MAX_DUPLICATE_RATE,
                "observed": round(duplicate_rate, 6),
                "threshold": MAX_DUPLICATE_RATE,
            },
            {"name": "target_cardinality", "passed": classes >= 2, "observed": classes, "threshold": 2},
            {"name": "numeric_features", "passed": numeric},
        ]
        if iris:
            checks.append(
                {
                    "name": "positive_measurements",
                    "passed": numeric and bool((data[features] > 0).all().all()),
                }
            )
    passed = sum(bool(item["passed"]) for item in checks)
    score = round(passed / len(checks), 4)
    return QualityResult(passed == len(checks), score, checks, profile_dataframe(frame))
