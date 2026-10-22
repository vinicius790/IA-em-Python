from __future__ import annotations

from typing import Any

DRIFT_PENALTY = {"warning": 15, "drift": 35}
QUALITY_PENALTY = 30
SLO_PENALTY = 15


def score_health(*, quality_passed: bool, drift_status: str, failed_slos: list[str]) -> dict[str, Any]:
    """Nota de 0 a 100 a partir de sinais medidos; cada desconto vem identificado."""
    penalties: list[dict[str, Any]] = []
    if not quality_passed:
        penalties.append({"reason": "dataset_quality_failed", "points": QUALITY_PENALTY})
    if drift_status in DRIFT_PENALTY:
        penalties.append({"reason": f"drift_{drift_status}", "points": DRIFT_PENALTY[drift_status]})
    penalties += [{"reason": f"slo_failed:{name}", "points": SLO_PENALTY} for name in failed_slos]
    score = max(0.0, 100.0 - sum(item["points"] for item in penalties))
    status = "healthy" if score >= 80 else "degraded" if score >= 50 else "critical"
    return {"score": score, "status": status, "penalties": penalties}
