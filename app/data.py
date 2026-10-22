from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import load_iris

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COLUMN = "class_name"


def load_iris_dataframe() -> pd.DataFrame:
    """Carrega o Iris do scikit-learn, sem depender de download."""
    data = load_iris(as_frame=True)
    frame = data.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )
    frame[TARGET_COLUMN] = frame["target"].map(dict(enumerate(data.target_names)))
    return frame[[*FEATURE_COLUMNS, TARGET_COLUMN]].copy()


def ensure_local_dataset(path: str | Path) -> Path:
    """Garante uma cópia CSV do dataset em `path` e devolve o caminho."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        load_iris_dataframe().to_csv(path, index=False)
    return path


def dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "features": FEATURE_COLUMNS.copy(),
        "classes": {str(k): int(v) for k, v in frame[TARGET_COLUMN].value_counts().sort_index().items()},
    }
