"""Intervalo de confiança da média pela distribuição t de Student (só biblioteca padrão)."""

from __future__ import annotations

import math
import statistics
from typing import Any

# Valores críticos bilaterais de t para 1 a 30 graus de liberdade.
# ponytail: tabela fixa para 90%, 95% e 99%; acima de 30 graus de liberdade usa a normal.
# Para outros níveis de confiança, troque por `scipy.stats.t.ppf`.
# fmt: off
_T_CRITICAL: dict[float, tuple[float, ...]] = {
    0.90: (
        6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812,
        1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725,
        1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697,
    ),
    0.95: (
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
    ),
    0.99: (
        63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250, 3.169,
        3.106, 3.055, 3.012, 2.977, 2.947, 2.921, 2.898, 2.878, 2.861, 2.845,
        2.831, 2.819, 2.807, 2.797, 2.787, 2.779, 2.771, 2.763, 2.756, 2.750,
    ),
}
# fmt: on
CONFIDENCE_LEVELS = tuple(_T_CRITICAL)


def t_critical(degrees_of_freedom: int, confidence: float = 0.95) -> float:
    if confidence not in _T_CRITICAL:
        raise ValueError(f"confidence must be one of: {', '.join(map(str, CONFIDENCE_LEVELS))}")
    if degrees_of_freedom < 1:
        raise ValueError("at least two observations are required")
    table = _T_CRITICAL[confidence]
    if degrees_of_freedom <= len(table):
        return table[degrees_of_freedom - 1]
    return statistics.NormalDist().inv_cdf(0.5 + confidence / 2)


def confidence_interval(values: list[float], confidence: float = 0.95) -> dict[str, Any]:
    """Média, desvio padrão amostral e intervalo de confiança t da média."""
    values = [float(value) for value in values]
    critical = t_critical(len(values) - 1, confidence)
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    margin = critical * std / math.sqrt(len(values))
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "lower": round(mean - margin, 6),
        "upper": round(mean + margin, 6),
        "confidence": confidence,
        "n": len(values),
    }
