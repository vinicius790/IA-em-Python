from __future__ import annotations

from typing import Any

import pandas as pd

from ..data import FEATURE_COLUMNS

ENGINEERED_COLUMNS = [
    "sepal_area",
    "petal_area",
    "sepal_aspect_ratio",
    "petal_aspect_ratio",
    "petal_to_sepal_length_ratio",
]


def enrich_iris(frame: Any) -> pd.DataFrame:
    """Acrescenta áreas e razões às quatro medidas originais.

    Aceita DataFrame ou matriz na ordem de `FEATURE_COLUMNS`. Estas colunas são usadas
    na análise e seleção de features; os modelos são treinados com as quatro medidas originais.
    """
    out = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame, columns=FEATURE_COLUMNS)
    out["sepal_area"] = out.sepal_length * out.sepal_width
    out["petal_area"] = out.petal_length * out.petal_width
    out["sepal_aspect_ratio"] = out.sepal_length / out.sepal_width.clip(lower=1e-9)
    out["petal_aspect_ratio"] = out.petal_length / out.petal_width.clip(lower=1e-9)
    out["petal_to_sepal_length_ratio"] = out.petal_length / out.sepal_length.clip(lower=1e-9)
    return out


def feature_manifest(frame: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """Contrato observável das colunas: papel, tipo e faixa de valores."""
    features = []
    for name in frame.columns:
        series = frame[name]
        item: dict[str, Any] = {
            "name": name,
            "dtype": str(series.dtype),
            "role": "target" if name == target else "feature",
            "engineered": name in ENGINEERED_COLUMNS,
        }
        if pd.api.types.is_numeric_dtype(series):
            item |= {"min": float(series.min()), "max": float(series.max()), "mean": float(series.mean())}
        features.append(item)
    return {"rows": len(frame), "target": target, "features": features}
