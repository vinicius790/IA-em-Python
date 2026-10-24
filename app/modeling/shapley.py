"""API simples de explicabilidade (antes em ``app/explainability.py``).

Importância por permutação (sklearn) e Shapley exato por coalizões.
"""

from __future__ import annotations

import itertools
from math import factorial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

MAX_SHAPLEY_FEATURES = 10


def permutation_explanation(model: Any, X: pd.DataFrame, y: Any, seed: int = 42) -> list[dict[str, Any]]:
    """Importância global: queda de acurácia ao embaralhar cada feature."""
    result = permutation_importance(model, X, y, scoring="accuracy", n_repeats=10, random_state=seed)
    ranking = [
        {"feature": feature, "importance_mean": round(float(mean), 8), "importance_std": round(float(std), 8)}
        for feature, mean, std in zip(X.columns, result.importances_mean, result.importances_std, strict=True)
    ]
    return sorted(ranking, key=lambda item: item["importance_mean"], reverse=True)


def shapley_explanation(
    model: Any, background: pd.DataFrame, instance: list[float], class_index: int
) -> dict[str, Any]:
    """Valores de Shapley exatos da instância para a probabilidade de uma classe.

    Para cada coalizão S de features, v(S) é a probabilidade média com as features
    de S fixas nos valores da instância e as demais vindas do `background`. Com n
    features são 2**n coalizões (16 no Iris), todas avaliadas numa única predição.
    A soma das contribuições é exatamente `prediction - base_value`.
    """
    features = list(background.columns)
    n = len(features)
    if n > MAX_SHAPLEY_FEATURES:
        raise ValueError(f"Exact Shapley supports at most {MAX_SHAPLEY_FEATURES} features")
    base = background.to_numpy(dtype=float)
    x = np.asarray(instance, dtype=float)
    coalitions = [frozenset(c) for size in range(n + 1) for c in itertools.combinations(range(n), size)]
    blocks = []
    for coalition in coalitions:
        block = base.copy()
        block[:, list(coalition)] = x[list(coalition)]
        blocks.append(block)
    probabilities = model.predict_proba(pd.DataFrame(np.vstack(blocks), columns=features))[:, class_index]
    means = probabilities.reshape(len(coalitions), len(base)).mean(axis=1)
    value = dict(zip(coalitions, means, strict=True))

    contributions = []
    for i, feature in enumerate(features):
        phi = 0.0
        for coalition in coalitions:
            if i in coalition:
                continue
            weight = factorial(len(coalition)) * factorial(n - len(coalition) - 1) / factorial(n)
            phi += weight * (value[coalition | {i}] - value[coalition])
        contributions.append({"feature": feature, "value": float(x[i]), "shap_value": round(float(phi), 8)})
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    return {
        "method": "exact_shapley",
        "base_value": round(float(value[frozenset()]), 8),
        "prediction": round(float(value[frozenset(range(n))]), 8),
        "background_rows": len(base),
        "contributions": contributions,
    }
