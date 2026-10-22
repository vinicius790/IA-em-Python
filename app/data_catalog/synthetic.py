from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.datasets import make_classification, make_regression


@dataclass(slots=True)
class SyntheticSpec:
    task: str = "classification"
    rows: int = 1000
    features: int = 10
    informative: int = 6
    classes: int = 3
    noise: float = 0.15
    random_state: int = 42

    def validate(self) -> None:
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        if self.rows < 10 or self.features < 2 or self.informative < 1:
            raise ValueError("rows>=10, features>=2 and informative>=1 are required")


def generate(spec: SyntheticSpec | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = spec or SyntheticSpec()
    spec.validate()
    if spec.informative > spec.features:
        raise ValueError("informative cannot exceed features")
    if spec.task == "classification":
        X, y = make_classification(
            n_samples=spec.rows,
            n_features=spec.features,
            n_informative=spec.informative,
            n_redundant=max(0, min(2, spec.features - spec.informative)),
            n_classes=spec.classes,
            random_state=spec.random_state,
        )
    else:
        X, y = make_regression(
            n_samples=spec.rows,
            n_features=spec.features,
            n_informative=spec.informative,
            noise=spec.noise,
            random_state=spec.random_state,
        )
    columns = [f"feature_{i:03d}" for i in range(spec.features)]
    frame = pd.DataFrame(X, columns=columns)
    target = "target"
    frame[target] = y
    return frame, {"spec": asdict(spec), "target": target, "columns": columns}
