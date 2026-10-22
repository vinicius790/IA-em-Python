from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from typing import Any


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "created_at": datetime.now(UTC).isoformat(),
    }


def dataset_fingerprint(frame) -> str:
    payload = frame.sort_index(axis=1).to_csv(index=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def lineage_record(
    *, model_name: str, dataset_hash: str, parameters: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    record = {
        "model_name": model_name,
        "dataset_hash": dataset_hash,
        "parameters": parameters,
        "metrics": metrics,
        "environment": environment_snapshot(),
    }
    record["lineage_hash"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, default=str).encode()
    ).hexdigest()
    return record
