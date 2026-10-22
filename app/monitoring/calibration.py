from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def multiclass_brier(y_true: Sequence[str], probabilities: np.ndarray, classes: Sequence[str]) -> float:
    """Brier score multiclasse: erro quadrático médio entre probabilidades e o one-hot real.

    0 é perfeito; 2 é o pior caso (certeza total na classe errada).
    """
    probabilities = np.asarray(probabilities, dtype=float)
    index = {label: position for position, label in enumerate(classes)}
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), [index[label] for label in y_true]] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
