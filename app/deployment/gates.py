from __future__ import annotations

from typing import Any


def evaluate_release_gate(
    metrics: dict[str, Any],
    *,
    min_accuracy: float = 0.90,
    min_f1: float = 0.90,
    max_latency_ms: float = 100.0,
) -> dict[str, Any]:
    """Compara as métricas medidas no treino com os limites; métrica ausente reprova."""

    def check(name: str, threshold: float, *, higher_is_better: bool) -> dict[str, Any]:
        observed = metrics.get(name)
        if not isinstance(observed, int | float):
            return {"name": name, "passed": False, "observed": None, "threshold": threshold}
        passed = observed >= threshold if higher_is_better else observed <= threshold
        return {"name": name, "passed": bool(passed), "observed": observed, "threshold": threshold}

    checks = [
        check("accuracy", min_accuracy, higher_is_better=True),
        check("f1_macro", min_f1, higher_is_better=True),
        check("latency_ms", max_latency_ms, higher_is_better=False),
    ]
    return {"passed": all(item["passed"] for item in checks), "checks": checks}
