"""Testes automatizados do Módulo de Explicabilidade Avançada de Modelos (XAI)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.data import FEATURE_COLUMNS, load_iris_dataframe
from app.modeling.explainability import (
    LocalPredictionDecomposition,
    compute_kernel_shap,
    pure_permutation_importance,
)


@pytest.fixture(scope="module")
def iris_model():
    df = load_iris_dataframe()
    X = df[FEATURE_COLUMNS]
    y = (df["class_name"] == "setosa").astype(int)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, X, y


def test_kernel_shap_additive_reconciliation_and_formatting(iris_model):
    model, X, y = iris_model
    background = X.sample(30, random_state=42)
    # Instância típica de setosa
    instance = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}

    decomp = compute_kernel_shap(
        model=model,
        background=background,
        instance=instance,
        feature_names=FEATURE_COLUMNS,
        target_class=1,
    )

    assert isinstance(decomp, LocalPredictionDecomposition)
    assert 0.0 <= decomp.base_value <= 1.0
    assert 0.0 <= decomp.prediction <= 1.0

    # Propriedade aditiva local fundamental de Shapley:
    # base_value + sum(contributions) == prediction
    total_contribution = sum(c.contribution for c in decomp.contributions)
    assert decomp.base_value + total_contribution == pytest.approx(decomp.prediction, abs=1e-4)

    # Formatação legível obrigatória: 'feature_name (+0.XX), ...'
    assert isinstance(decomp.formatted_explanation, str)
    assert len(decomp.formatted_explanation) > 0
    assert "petal_length" in decomp.formatted_explanation
    # Ordenação por magnitude absoluta de impacto
    contributions = decomp.contributions
    for i in range(len(contributions) - 1):
        assert contributions[i].abs_contribution >= contributions[i + 1].abs_contribution

    # Estrutura do waterfall
    assert decomp.waterfall_steps[0]["step"] == "base_value"
    assert decomp.waterfall_steps[-1]["step"] == "final_prediction"


def test_kernel_shap_dict_serialization(iris_model):
    model, X, _ = iris_model
    background = X.head(20)
    instance = [6.0, 3.0, 4.8, 1.8]

    decomp = compute_kernel_shap(
        model=model,
        background=background,
        instance=instance,
        feature_names=FEATURE_COLUMNS,
        target_class=0,
    )
    d = decomp.to_dict()

    assert "base_value" in d
    assert "prediction" in d
    assert "reconciled_prediction" in d
    assert "contributions" in d
    assert "formatted_explanation" in d
    assert "waterfall_steps" in d
    assert len(d["contributions"]) == 4


def test_pure_permutation_importance(iris_model):
    model, X, y = iris_model

    ranking = pure_permutation_importance(
        model=model,
        X=X,
        y=y,
        n_repeats=3,
        seed=42,
    )

    assert len(ranking) == 4
    for item in ranking:
        assert "feature" in item
        assert "importance_mean" in item
        assert "importance_std" in item
        assert item["feature"] in FEATURE_COLUMNS

    # O ranking deve vir ordenado de forma decrescente pela importância
    for i in range(len(ranking) - 1):
        assert ranking[i]["importance_mean"] >= ranking[i + 1]["importance_mean"]
