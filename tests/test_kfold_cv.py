"""Testes do Motor de Validação Cruzada K-Fold com Intervalos de Confiança."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.data import FEATURE_COLUMNS, TARGET_COLUMN, load_iris_dataframe
from app.modeling.cross_validation import KFoldValidationResult, run_kfold_cross_validation


@pytest.fixture(scope="module")
def iris_data():
    df = load_iris_dataframe()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def test_kfold_cross_validation_stratified(iris_data):
    X, y = iris_data
    model = LogisticRegression(max_iter=300, random_state=42)

    res = run_kfold_cross_validation(
        model=model,
        X=X,
        y=y,
        model_name="logistic_regression",
        n_splits=5,
        stratified=True,
        random_state=42,
    )

    assert isinstance(res, KFoldValidationResult)
    assert res.n_splits == 5
    assert res.stratified is True
    assert res.sample_count == 150
    assert len(res.classes) == 3
    assert len(res.fold_metrics) == 5

    # Cada fold deve ter métricas consistentes
    for fold in res.fold_metrics:
        assert 0.80 <= fold["accuracy"] <= 1.0
        assert 0.80 <= fold["f1_macro"] <= 1.0
        assert fold["train_samples"] + fold["val_samples"] == 150

    # Agregados
    agg = res.aggregated_metrics
    assert "accuracy" in agg
    assert "f1_macro" in agg
    assert 0.85 <= agg["accuracy"]["mean"] <= 1.0
    assert agg["accuracy"]["ci_lower"] <= agg["accuracy"]["mean"] <= agg["accuracy"]["ci_upper"]

    # Matriz de confusão out-of-fold: soma total deve ser igual a sample_count
    cm = np.array(res.confusion_matrix)
    assert cm.shape == (3, 3)
    assert cm.sum() == 150

    # Serialização para dict
    d = res.to_dict()
    assert d["model_name"] == "logistic_regression"
    assert len(d["fold_metrics"]) == 5


def test_kfold_cross_validation_invalid_splits(iris_data):
    X, y = iris_data
    model = LogisticRegression(max_iter=200)

    with pytest.raises(ValueError, match="n_splits deve ser pelo menos 2"):
        run_kfold_cross_validation(model, X, y, n_splits=1)

    with pytest.raises(ValueError, match="não pode ser maior"):
        run_kfold_cross_validation(model, X.head(5), y.head(5), n_splits=10)
