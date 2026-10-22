from __future__ import annotations

from typing import Any

import pandas as pd


def correlations(frame: pd.DataFrame, method: str = "pearson") -> dict[str, Any]:
    """Matriz de correlação das colunas numéricas."""
    numeric = frame.select_dtypes(include="number")
    return {"method": method, "matrix": numeric.corr(method=method).round(6).to_dict()}
