"""Regras do deploy canário: divisão de tráfego, métricas por braço e decisão.

Só biblioteca padrão. Quem persiste e serve é `app/services.py`; aqui ficam as
funções puras, fáceis de testar.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import asdict, dataclass
from typing import Any

PROMOTE, ROLLBACK, WAIT = "promote", "rollback", "wait"
# Predições mais recentes de cada braço que entram nas métricas. `min_requests` e
# `min_labeled` não podem passar disso, senão a regra esperaria para sempre.
ARM_WINDOW = 500


@dataclass(frozen=True, slots=True)
class CanaryRule:
    """Limites da decisão automática.

    - `min_requests`: predições da candidata antes de qualquer decisão.
    - `max_confidence_drop`: quanto a confiança média da candidata pode ficar abaixo da estável.
    - `max_latency_ratio`: p95 da candidata dividido pelo p95 da estável.
    - `max_accuracy_drop`: queda de acurácia tolerada, quando há pelo menos `min_labeled`
      predições rotuladas (feedback) em cada braço.
    """

    min_requests: int = 50
    max_confidence_drop: float = 0.05
    max_latency_ratio: float = 1.5
    max_accuracy_drop: float = 0.02
    min_labeled: int = 20

    def __post_init__(self) -> None:
        if not 1 <= self.min_requests <= ARM_WINDOW or not 1 <= self.min_labeled <= ARM_WINDOW:
            raise ValueError(f"min_requests and min_labeled must be between 1 and {ARM_WINDOW}")
        if not 0 <= self.max_confidence_drop <= 1 or not 0 <= self.max_accuracy_drop <= 1:
            raise ValueError("max_confidence_drop and max_accuracy_drop must be between 0 and 1")
        if self.max_latency_ratio < 1:
            raise ValueError("max_latency_ratio must be >= 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_arm(routing_key: str, traffic_percent: float) -> str:
    """`candidate` para `traffic_percent`% das chaves; determinístico por chave.

    A mesma chave (ex.: o X-Request-ID, ou um id de cliente) cai sempre no mesmo braço,
    o que mantém a experiência do cliente estável durante o canário.
    """
    if not 0 <= traffic_percent <= 100:
        raise ValueError("traffic_percent must be between 0 and 100")
    bucket = int(hashlib.sha256(routing_key.encode()).hexdigest()[:8], 16) % 10_000
    return "candidate" if bucket < traffic_percent * 100 else "stable"


def validate_canary_attribution(canary_id: int | None, arm: str | None) -> tuple[int | None, str | None]:
    """Valida o contexto opcional gravado com uma predição.

    Predições fora de um experimento não recebem atribuição. Dentro dele, id e braço
    são inseparáveis para que relatórios não misturem tráfego comum com tráfego canário.
    """
    if canary_id is None and arm is None:
        return None, None
    if isinstance(canary_id, bool) or not isinstance(canary_id, int) or canary_id < 1:
        raise ValueError("canary_id must be a positive integer when an arm is provided")
    if arm not in {"stable", "candidate"}:
        raise ValueError("arm must be 'stable' or 'candidate' when canary_id is provided")
    return canary_id, arm


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def arm_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Resumo de um braço; cada linha traz `confidence`, `latency_ms`, `predicted_class` e `actual_class`."""
    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    labeled = [row for row in rows if row.get("actual_class")]
    return {
        "requests": len(rows),
        "mean_confidence": round(statistics.fmean(row["confidence"] for row in rows), 6) if rows else None,
        "p95_latency_ms": round(_p95(latencies), 4) if latencies else None,
        "labeled": len(labeled),
        "accuracy": (
            round(sum(row["predicted_class"] == row["actual_class"] for row in labeled) / len(labeled), 6)
            if labeled
            else None
        ),
    }


def decide(stable: dict[str, Any], candidate: dict[str, Any], rule: CanaryRule) -> dict[str, Any]:
    """Decide entre promover, reverter ou esperar, explicando cada verificação.

    Com a estável sem tráfego suficiente para comparar, só as verificações que têm os
    dois lados são aplicadas; a candidata nunca é promovida sem `min_requests`.
    """
    if candidate["requests"] < rule.min_requests:
        return {
            "decision": WAIT,
            "reason": f"candidate has {candidate['requests']} of {rule.min_requests} required requests",
            "checks": [],
        }
    checks: list[dict[str, Any]] = []
    if stable.get("mean_confidence") is not None:
        drop = stable["mean_confidence"] - candidate["mean_confidence"]
        checks.append(
            {
                "name": "confidence",
                "passed": drop <= rule.max_confidence_drop,
                "observed": round(drop, 6),
                "threshold": rule.max_confidence_drop,
            }
        )
    if stable.get("p95_latency_ms") and candidate.get("p95_latency_ms") is not None:
        ratio = candidate["p95_latency_ms"] / stable["p95_latency_ms"]
        checks.append(
            {
                "name": "latency",
                "passed": ratio <= rule.max_latency_ratio,
                "observed": round(ratio, 4),
                "threshold": rule.max_latency_ratio,
            }
        )
    if stable["labeled"] >= rule.min_labeled and candidate["labeled"] >= rule.min_labeled:
        drop = stable["accuracy"] - candidate["accuracy"]
        checks.append(
            {
                "name": "accuracy",
                "passed": drop <= rule.max_accuracy_drop,
                "observed": round(drop, 6),
                "threshold": rule.max_accuracy_drop,
            }
        )
    failed = [check["name"] for check in checks if not check["passed"]]
    if failed:
        return {"decision": ROLLBACK, "reason": f"candidate failed: {', '.join(failed)}", "checks": checks}
    return {"decision": PROMOTE, "reason": "candidate met every canary check", "checks": checks}
