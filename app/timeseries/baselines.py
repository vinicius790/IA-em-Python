from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ForecastResult:
    method: str
    horizon: int
    values: list[float]


def forecast(values: list[float], horizon: int, method: str = "naive", window: int = 5) -> ForecastResult:
    if not values or horizon < 1:
        raise ValueError("values must be non-empty and horizon >= 1")
    x = np.asarray(values, dtype=float)
    if method == "naive":
        pred = np.repeat(x[-1], horizon)
    elif method == "moving_average":
        if window < 1:
            raise ValueError("window >= 1")
        pred = np.repeat(float(x[-min(window, len(x)) :].mean()), horizon)
    elif method == "linear_trend":
        n = len(x)
        coef = np.polyfit(np.arange(n), x, 1)
        pred = np.polyval(coef, np.arange(n, n + horizon))
    else:
        raise ValueError(f"Unknown method: {method}")
    return ForecastResult(method, horizon, [round(float(v), 8) for v in pred])


def mae(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("vectors must have equal non-zero length")
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def backtest(
    values: list[float], horizon: int = 1, method: str = "naive", min_train: int = 8, window: int = 5
) -> dict[str, Any]:
    """Backtest de origem deslizante: reprevê a cada ponto a partir de `min_train` observações."""
    if len(values) <= min_train + horizon:
        raise ValueError("not enough observations")
    errors = []
    for end in range(min_train, len(values) - horizon + 1):
        prediction = forecast(values[:end], horizon, method, window).values
        actual = values[end : end + horizon]
        errors.append(mae(actual, prediction))
    return {
        "method": method,
        "horizon": horizon,
        "windows": len(errors),
        "mae": round(float(np.mean(errors)), 8),
    }
