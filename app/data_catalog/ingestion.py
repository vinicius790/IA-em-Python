from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SUPPORTED = {".csv", ".json", ".jsonl", ".parquet"}


def ingest(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported format: {p.suffix}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".jsonl":
        return pd.read_json(p, lines=True)
    data = json.loads(p.read_text(encoding="utf-8"))
    return pd.DataFrame(data)


def write_dataset(frame: pd.DataFrame, path: str | Path, format: str = "csv") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if format == "csv":
        frame.to_csv(p, index=False)
    elif format == "parquet":
        frame.to_parquet(p, index=False)
    elif format == "json":
        p.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported output format: {format}")
    return p
