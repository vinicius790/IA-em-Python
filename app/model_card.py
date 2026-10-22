from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def create_model_card(
    model_name: str,
    metrics: dict[str, Any],
    parameters: dict[str, Any],
    classes: list[str],
    dataset: str = "iris",
) -> dict[str, Any]:
    if dataset != "iris":
        return {
            "model_name": model_name,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": dataset,
            "intended_use": f"Multi-class classification on the uploaded dataset '{dataset}'.",
            "limitations": [
                "Trained on a user-uploaded CSV; its representativeness was not assessed by the platform.",
                "Metrics are dataset-specific and should not be generalized to other populations.",
            ],
            "classes": classes,
            "metrics": metrics,
            "training_parameters": parameters,
            "monitoring": {"recommended": ["data drift", "latency", "confidence", "class distribution"]},
        }
    return {
        "model_name": model_name,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "intended_use": "Educational and demonstration multi-class Iris classification.",
        "limitations": [
            "Iris is a tiny canonical dataset and does not represent production-scale drift or deployment risk.",
            "Metrics are dataset-specific and should not be generalized to other populations.",
        ],
        "classes": classes,
        "metrics": metrics,
        "training_parameters": parameters,
        "monitoring": {"recommended": ["data drift", "latency", "confidence", "class distribution"]},
    }
