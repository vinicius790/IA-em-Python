from __future__ import annotations

import os
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

PACKAGES = [
    "fastapi",
    "pydantic",
    "SQLAlchemy",
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "joblib",
    "typer",
    "prometheus-client",
]


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def runtime_snapshot() -> dict[str, Any]:
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
    }
