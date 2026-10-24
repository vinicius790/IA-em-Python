"""Explicabilidade Avançada de Modelos (XAI - Explainable AI).

Implementação pura em Python/NumPy:
- Kernel SHAP simplificado e Shapley Exato para atribuição aditiva de features.
- Decomposição local de predições com formatação explícita (ex: 'petal_length (+0.42), sepal_width (-0.15)').
- Importância por permutação pura (Permutation Feature Importance) sem dependências binárias pesadas.
- Gráficos waterfall conceituais e reconciliação aditiva: sum(phi_i) = f(x) - E[f(X)].
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from math import factorial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


@dataclass
class LocalFeatureContribution:
    """Contribuição local de uma feature para a decisão de uma instância."""

    feature: str
    feature_value: float
    contribution: float  # Valor de Shapley / Phi
    abs_contribution: float
    direction: str  # "positive", "negative", "neutral"
    formatted_str: str  # ex: "petal_length (+0.42)"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocalPredictionDecomposition:
    """Decomposição aditiva da predição local do modelo."""

    instance: Dict[str, float]
    base_value: float  # Valor esperado E[f(X)] sobre o background
    prediction: float  # Predição f(x) para a instância
    reconciled_prediction: float  # base_value + sum(contributions)
    target_class: Optional[Union[str, int]]
    contributions: List[LocalFeatureContribution]
    formatted_explanation: str  # ex: 'petal_length (+0.42), sepal_width (-0.15)'
    waterfall_steps: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance": self.instance,
            "base_value": round(self.base_value, 4),
            "prediction": round(self.prediction, 4),
            "reconciled_prediction": round(self.reconciled_prediction, 4),
            "target_class": self.target_class,
            "contributions": [c.to_dict() for c in self.contributions],
            "formatted_explanation": self.formatted_explanation,
            "waterfall_steps": self.waterfall_steps,
        }


def _predict_output(model: Any, X: Union[pd.DataFrame, np.ndarray], target_class: Optional[int] = None) -> np.ndarray:
    """Obtém as saídas do modelo (probabilidade da classe ou valor contínuo)."""
    if hasattr(model, "predict_proba") and target_class is not None:
        probs = model.predict_proba(X)
        return probs[:, target_class] if probs.ndim == 2 else probs
    elif hasattr(model, "predict"):
        preds = model.predict(X)
        return preds.astype(float)
    elif callable(model):
        res = model(X)
        return np.asarray(res, dtype=float)
    raise AttributeError("O modelo deve implementar `predict_proba`, `predict` ou ser um callable.")


def compute_kernel_shap(
    model: Any,
    background: pd.DataFrame,
    instance: Union[Dict[str, float], Sequence[float], pd.Series],
    feature_names: Optional[List[str]] = None,
    target_class: Optional[int] = None,
    n_samples: int = 128,
    seed: int = 42,
) -> LocalPredictionDecomposition:
    """Calcula valores de Shapley via Kernel SHAP ou Shapley Exato.

    Garante a propriedade aditiva fundamental:
    prediction = base_value + sum(contributions).
    """
    if isinstance(instance, pd.Series):
        names = list(instance.index)
        x_vec = instance.to_numpy(dtype=float)
    elif isinstance(instance, dict):
        names = list(instance.keys())
        x_vec = np.array([float(instance[k]) for k in names])
    else:
        names = feature_names or [f"feature_{i}" for i in range(len(instance))]
        x_vec = np.asarray(instance, dtype=float)

    n_features = len(names)
    bg_df = background[names] if names[0] in background.columns else background
    bg_mat = bg_df.to_numpy(dtype=float)

    # 1. Base Value: E[f(X)]
    bg_preds = _predict_output(model, pd.DataFrame(bg_mat, columns=names), target_class=target_class)
    base_value = float(np.mean(bg_preds))

    # 2. Predição da instância f(x)
    x_df = pd.DataFrame([x_vec], columns=names)
    instance_pred = float(_predict_output(model, x_df, target_class=target_class)[0])

    # Se n_features <= 10, calcula Shapley exato combinatório
    if n_features <= 10:
        coalitions = [frozenset(c) for r in range(n_features + 1) for c in itertools.combinations(range(n_features), r)]
        blocks = []
        for c in coalitions:
            block = bg_mat.copy()
            if len(c) > 0:
                block[:, list(c)] = x_vec[list(c)]
            blocks.append(block)

        stacked = np.vstack(blocks)
        all_preds = _predict_output(model, pd.DataFrame(stacked, columns=names), target_class=target_class)
        means = all_preds.reshape(len(coalitions), len(bg_mat)).mean(axis=1)
        val_map = dict(zip(coalitions, means, strict=True))

        phi = np.zeros(n_features)
        for i in range(n_features):
            for c in coalitions:
                if i in c:
                    continue
                size = len(c)
                weight = (factorial(size) * factorial(n_features - size - 1)) / factorial(n_features)
                phi[i] += weight * (val_map[c | {i}] - val_map[c])
    else:
        # Kernel SHAP amostrado via regressão linear ponderada pelo Shapley Kernel
        rng = np.random.default_rng(seed)
        # Amostragem de máscaras binárias z' em {0, 1}^M
        z_masks = rng.integers(0, 2, size=(n_samples, n_features))
        # Adicionar máscaras extremas (todas 0 e todas 1)
        z_masks = np.vstack([np.zeros((1, n_features), dtype=int), np.ones((1, n_features), dtype=int), z_masks])

        weights = []
        synthetic_evals = []
        for z in z_masks:
            active_count = np.sum(z)
            if active_count == 0 or active_count == n_features:
                w = 1e6  # Grande peso para forçar limites
            else:
                w = (n_features - 1) / (
                    itertools.comb(n_features, active_count) * active_count * (n_features - active_count)
                )
            weights.append(w)

            # Criar amostra combinada
            block = bg_mat.copy()
            if active_count > 0:
                block[:, z == 1] = x_vec[z == 1]
            p_mean = np.mean(_predict_output(model, pd.DataFrame(block, columns=names), target_class=target_class))
            synthetic_evals.append(p_mean)

        W = np.diag(weights)
        Z = z_masks
        y_adj = np.array(synthetic_evals) - base_value

        # Resolver WLS: (Z^T W Z)^-1 Z^T W y_adj
        try:
            phi = np.linalg.solve(Z.T @ W @ Z + 1e-4 * np.eye(n_features), Z.T @ W @ y_adj)
        except np.linalg.LinAlgError:
            phi = np.linalg.lstsq(np.sqrt(W) @ Z, np.sqrt(W) @ y_adj, rcond=None)[0]

    # Reconciliação aditiva para eliminar pequenos resíduos numéricos
    total_diff = instance_pred - base_value
    current_sum = float(np.sum(phi))
    if abs(current_sum) > 1e-8:
        phi = phi * (total_diff / current_sum)
    else:
        phi = np.full(n_features, total_diff / n_features)

    contributions: List[LocalFeatureContribution] = []
    formatted_parts: List[str] = []
    sorted_indices = np.argsort(-np.abs(phi))

    for idx in sorted_indices:
        fname = names[idx]
        val = float(x_vec[idx])
        c = float(phi[idx])
        sign_str = f"+{c:.2f}" if c >= 0 else f"{c:.2f}"
        direction = "positive" if c > 1e-4 else ("negative" if c < -1e-4 else "neutral")
        fmt = f"{fname} ({sign_str})"
        formatted_parts.append(fmt)

        contributions.append(
            LocalFeatureContribution(
                feature=fname,
                feature_value=round(val, 4),
                contribution=round(c, 4),
                abs_contribution=round(abs(c), 4),
                direction=direction,
                formatted_str=fmt,
            )
        )

    # Construção dos passos do Waterfall
    waterfall_steps = [{"step": "base_value", "value": round(base_value, 4), "cumulative": round(base_value, 4)}]
    cum = base_value
    for c in contributions:
        cum += c.contribution
        waterfall_steps.append(
            {
                "step": c.feature,
                "contribution": c.contribution,
                "cumulative": round(cum, 4),
            }
        )
    waterfall_steps.append({"step": "final_prediction", "value": round(instance_pred, 4), "cumulative": round(cum, 4)})

    formatted_explanation = ", ".join(formatted_parts)

    return LocalPredictionDecomposition(
        instance={names[i]: float(x_vec[i]) for i in range(n_features)},
        base_value=base_value,
        prediction=instance_pred,
        reconciled_prediction=base_value + float(np.sum(phi)),
        target_class=target_class,
        contributions=contributions,
        formatted_explanation=formatted_explanation,
        waterfall_steps=waterfall_steps,
    )


def pure_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: Union[Sequence[Any], np.ndarray, pd.Series],
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_repeats: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Calcula Permutation Feature Importance pura em NumPy sem dependências binárias.

    Mede o decréscimo de performance (ex: queda na acurácia) ao embaralhar cada coluna.
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y)

    if metric_fn is None:
        # Padrão: acurácia para classificação
        def default_acc(y_t: np.ndarray, y_p: np.ndarray) -> float:
            return float(np.mean(y_t == y_p))

        metric_fn = default_acc

    base_preds = model.predict(X)
    baseline_score = metric_fn(y_arr, np.asarray(base_preds))

    features = list(X.columns)
    results = []

    for col in features:
        scores = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[col] = rng.permutation(X_shuffled[col].values)
            shuffled_preds = model.predict(X_shuffled)
            score = metric_fn(y_arr, np.asarray(shuffled_preds))
            scores.append(baseline_score - score)

        mean_drop = float(np.mean(scores))
        std_drop = float(np.std(scores))

        results.append(
            {
                "feature": col,
                "importance_mean": round(mean_drop, 4),
                "importance_std": round(std_drop, 4),
            }
        )

    results.sort(key=lambda item: item["importance_mean"], reverse=True)
    return results


@dataclass
class ExplainabilityReport:
    """Relatório estruturado de explicabilidade global e local."""

    model_name: str
    feature_names: List[str]
    permutation_importance: List[Dict[str, Any]]
    model_feature_importance: Optional[List[Dict[str, Any]]]
    sample_explanations: List[Dict[str, Any]]
    top_features: List[str]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExplainabilityPipeline:
    """Pipeline completo de Explicabilidade de Modelos de Machine Learning (XAI).

    Integra:
    - Feature Importance nativa do algoritmo (Tree gain/weight ou Coeficientes lineares).
    - Permutation Feature Importance em validação com média e desvio padrão.
    - Explicabilidade local aditiva via Kernel SHAP / Shapley Exato.
    - Diagnóstico de interpretabilidade e sumarização para relatórios técnicos.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str = "model",
        background_data: Optional[pd.DataFrame] = None,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.model_name = model_name
        self.background_data = background_data

    def extract_model_importance(self) -> Optional[List[Dict[str, Any]]]:
        """Extrai importância nativa do modelo se suportado (árvores ou linear)."""
        estimator = self.model
        if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
            estimator = estimator.named_steps["model"]

        # Árvores: feature_importances_
        if hasattr(estimator, "feature_importances_"):
            raw_imp = estimator.feature_importances_
            total = float(np.sum(raw_imp)) if np.sum(raw_imp) > 0 else 1.0
            ranking = [
                {
                    "feature": fname,
                    "importance": round(float(imp), 4),
                    "relative_percentage": round(float(imp) / total * 100, 2),
                }
                for fname, imp in zip(self.feature_names, raw_imp, strict=False)
            ]
            ranking.sort(key=lambda item: item["importance"], reverse=True)
            return ranking

        # Modelos Lineares: coef_
        if hasattr(estimator, "coef_"):
            coefs = np.asarray(estimator.coef_)
            # Para multiclasse pega a magnitude média absoluta dos coeficientes
            if coefs.ndim > 1:
                mag = np.mean(np.abs(coefs), axis=0)
            else:
                mag = np.abs(coefs)
            total = float(np.sum(mag)) if np.sum(mag) > 0 else 1.0
            ranking = [
                {
                    "feature": fname,
                    "importance": round(float(val), 4),
                    "relative_percentage": round(float(val) / total * 100, 2),
                }
                for fname, val in zip(self.feature_names, mag, strict=False)
            ]
            ranking.sort(key=lambda item: item["importance"], reverse=True)
            return ranking

        return None

    def run_permutation_importance(
        self,
        X: pd.DataFrame,
        y: Union[Sequence[Any], np.ndarray, pd.Series],
        n_repeats: int = 5,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Calcula Permutation Importance no conjunto de dados avaliado."""
        return pure_permutation_importance(
            model=self.model,
            X=X[self.feature_names] if all(c in X.columns for c in self.feature_names) else X,
            y=y,
            n_repeats=n_repeats,
            seed=seed,
        )

    def explain_instance(
        self,
        instance: Union[Dict[str, float], Sequence[float]],
        target_class: Optional[int] = 0,
        n_samples: int = 128,
    ) -> LocalPredictionDecomposition:
        """Decompõe uma instância individual em contribuições de Shapley."""
        if self.background_data is None:
            raise ValueError("Background data é obrigatório para decomposição de instâncias locais.")
        bg = self.background_data[self.feature_names] if all(c in self.background_data.columns for c in self.feature_names) else self.background_data
        return compute_kernel_shap(
            model=self.model,
            background=bg,
            instance=instance,
            feature_names=self.feature_names,
            target_class=target_class,
            n_samples=n_samples,
        )

    def run_pipeline(
        self,
        X: pd.DataFrame,
        y: Union[Sequence[Any], np.ndarray, pd.Series],
        sample_instances: Optional[List[Dict[str, float]]] = None,
        n_repeats: int = 5,
        seed: int = 42,
    ) -> ExplainabilityReport:
        """Executa o pipeline completo e consolida o relatório de interpretabilidade."""
        perm_imp = self.run_permutation_importance(X, y, n_repeats=n_repeats, seed=seed)
        model_imp = self.extract_model_importance()

        bg = self.background_data if self.background_data is not None else X
        samples_decomp: List[Dict[str, Any]] = []

        if sample_instances:
            for inst in sample_instances:
                decomp = compute_kernel_shap(
                    model=self.model,
                    background=bg,
                    instance=inst,
                    feature_names=self.feature_names,
                    target_class=0,
                )
                samples_decomp.append(decomp.to_dict())

        top_features = [item["feature"] for item in perm_imp[:3]]
        summary = (
            f"Modelo '{self.model_name}' auditado via XAI. "
            f"Features mais determinantes por permutação: {', '.join(top_features)}."
        )

        return ExplainabilityReport(
            model_name=self.model_name,
            feature_names=self.feature_names,
            permutation_importance=perm_imp,
            model_feature_importance=model_imp,
            sample_explanations=samples_decomp,
            top_features=top_features,
            summary=summary,
        )


# ---------------------------------------------------------------- API simples (antes em app/explainability.py)
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
