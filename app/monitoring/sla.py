from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SLO:
    name: str
    target: float
    direction: str = "gte"


class SLOEvaluator:
    def __init__(self, slos: list[SLO]):
        self.slos = slos

    def evaluate(self, observed: dict[str, float]) -> dict[str, Any]:
        checks = []
        for slo in self.slos:
            if slo.name not in observed:
                checks.append({"name": slo.name, "status": "unknown", "target": slo.target})
                continue
            value = float(observed[slo.name])
            passed = value >= slo.target if slo.direction == "gte" else value <= slo.target
            checks.append(
                {
                    "name": slo.name,
                    "observed": value,
                    "target": slo.target,
                    "direction": slo.direction,
                    "passed": passed,
                }
            )
        evaluated = [row for row in checks if "passed" in row]
        return {"passed": all(row["passed"] for row in evaluated) if evaluated else False, "checks": checks}
