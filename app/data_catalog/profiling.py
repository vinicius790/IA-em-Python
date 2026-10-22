from __future__ import annotations

from typing import Any

import pandas as pd


def _semantic_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def profile_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    """Perfil por coluna: tipo, nulos, cardinalidade e estatísticas básicas."""
    columns = []
    for name in frame.columns:
        series = frame[name]
        semantic = _semantic_type(series)
        item: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "semantic_type": semantic,
            "nulls": int(series.isna().sum()),
            "null_rate": round(float(series.isna().mean()), 6) if len(series) else 0.0,
            "unique": int(series.nunique(dropna=True)),
        }
        if semantic == "numeric" and series.notna().any():
            item["numeric"] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": round(float(series.mean()), 8),
                "std": round(float(series.std(ddof=0)), 8),
                "p01": round(float(series.quantile(0.01)), 8),
                "p50": round(float(series.quantile(0.50)), 8),
                "p99": round(float(series.quantile(0.99)), 8),
            }
        elif semantic == "categorical":
            item["top_values"] = {
                str(k): int(v) for k, v in series.value_counts(dropna=True).head(10).items()
            }
        columns.append(item)
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        "column_profiles": columns,
    }
