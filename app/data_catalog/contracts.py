from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class ColumnContract:
    name: str
    dtype: str | None = None
    nullable: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None


@dataclass(slots=True)
class DatasetContract:
    name: str
    columns: list[ColumnContract]
    min_rows: int = 1
    target: str | None = None

    def validate(self, frame: pd.DataFrame) -> dict[str, Any]:
        checks = []
        for col in self.columns:
            present = col.name in frame.columns
            checks.append({"check": f"column:{col.name}", "passed": present})
            if not present:
                continue
            series = frame[col.name]
            if col.dtype:
                checks.append({"check": f"dtype:{col.name}", "passed": str(series.dtype) == col.dtype})
            if not col.nullable:
                checks.append({"check": f"nonnull:{col.name}", "passed": bool(series.notna().all())})
            if col.min_value is not None and pd.api.types.is_numeric_dtype(series):
                checks.append({"check": f"min:{col.name}", "passed": bool(series.min() >= col.min_value)})
            if col.max_value is not None and pd.api.types.is_numeric_dtype(series):
                checks.append({"check": f"max:{col.name}", "passed": bool(series.max() <= col.max_value)})
            if col.allowed_values is not None:
                checks.append(
                    {
                        "check": f"allowed:{col.name}",
                        "passed": set(series.dropna().astype(str).unique()) <= set(col.allowed_values),
                    }
                )
        checks.append({"check": "min_rows", "passed": len(frame) >= self.min_rows})
        return {
            "name": self.name,
            "passed": all(c["passed"] for c in checks),
            "rows": len(frame),
            "columns": list(frame.columns),
            "checks": checks,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "min_rows": self.min_rows,
            "columns": [asdict(c) for c in self.columns],
        }
