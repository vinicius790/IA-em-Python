from __future__ import annotations

from typing import Any


def leaderboard(records: list[dict[str, Any]], metric: str = "f1_macro") -> list[dict[str, Any]]:
    """Ordena os registros do registry por uma métrica; não altera estágios."""
    rows = []
    for record in records:
        value = record.get("metrics", {}).get(metric)
        rows.append(
            {
                "name": record.get("name"),
                "version": record.get("version"),
                "stage": record.get("stage"),
                "metric": metric,
                "value": float(value) if isinstance(value, int | float) else None,
                "created_at": record.get("created_at"),
            }
        )
    return sorted(rows, key=lambda row: (row["value"] is None, -(row["value"] or 0.0)))


def pareto_frontier(
    records: list[dict[str, Any]], maximize: str = "f1_macro", minimize: str = "latency_ms"
) -> list[dict[str, Any]]:
    """Registros não dominados no compromisso qualidade (maior é melhor) x latência (menor é melhor)."""

    def point(record: dict[str, Any]) -> tuple[float, float] | None:
        metrics = record.get("metrics") or {}
        quality, cost = metrics.get(maximize), metrics.get(minimize)
        if isinstance(quality, int | float) and isinstance(cost, int | float):
            return float(quality), -float(cost)
        return None

    points = [(record, point(record)) for record in records]
    measured = [(record, p) for record, p in points if p is not None]
    return [
        record
        for record, p in measured
        if not any(q[0] >= p[0] and q[1] >= p[1] and q != p for _, q in measured)
    ]
