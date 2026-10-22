from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

OPTIONAL_PACKAGES = ("optuna", "lightgbm", "catboost", "mlflow")


def optional_stack_status() -> dict[str, Any]:
    """Quais dependências opcionais (extras `tuning`, `boosters`, `tracking`) estão instaladas."""
    result: dict[str, Any] = {}
    for name in OPTIONAL_PACKAGES:
        try:
            result[name] = {"installed": True, "version": version(name)}
        except PackageNotFoundError:
            result[name] = {"installed": False, "version": None}
    return result
