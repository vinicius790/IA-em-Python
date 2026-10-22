"""Testes do Pipeline Integrado de Explicabilidade (XAI) e Feature Importance."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.data import FEATURE_COLUMNS, TARGET_COLUMN, load_iris_dataframe
from app.modeling.explainability import (
    ExplainabilityPipeline,
    ExplainabilityReport,
    LocalPredictionDecomposition,
)


@pytest.fixture(scope="module")
def iris_dataset():
    df = load_iris_dataframe()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def test_explainability_pipeline_linear_model(iris_dataset):
    X, y = iris_dataset
    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(X, y)

    pipeline = ExplainabilityPipeline(
        model=model,
        feature_names=FEATURE_COLUMNS,
        model_name="logistic_regression",
        background_data=X,
    )

    # 1. Importância nativa baseada em coeficientes
    model_imp = pipeline.extract_model_importance()
    assert model_imp is not None
    assert len(model_imp) == 4
    assert all("feature" in item and "importance" in item for item in model_imp)
    # Soma percentual deve ser ~100%
    total_pct = sum(item["relative_percentage"] for item in model_imp)
    assert total_pct == pytest.approx(100.0, abs=0.5)

    # 2. Permutation Importance
    perm_imp = pipeline.run_permutation_importance(X, y, n_repeats=3, seed=42)
    assert len(perm_imp) == 4
    for item in perm_imp:
        assert "feature" in item
        assert "importance_mean" in item
        assert "importance_std" in item

    # 3. Explicabilidade local de instância
    sample = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    decomp = pipeline.explain_instance(sample, target_class=0)
    assert isinstance(decomp, LocalPredictionDecomposition)
    assert len(decomp.contributions) == 4

    # 4. Relatório consolidado
    report = pipeline.run_pipeline(X, y, sample_instances=[sample], n_repeats=3)
    assert isinstance(report, ExplainabilityReport)
    assert report.model_name == "logistic_regression"
    assert len(report.top_features) > 0
    assert len(report.sample_explanations) == 1
    d = report.to_dict()
    assert "summary" in d
    assert "permutation_importance" in d


def test_explainability_pipeline_tree_model(iris_dataset):
    X, y = iris_dataset
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    pipeline = ExplainabilityPipeline(
        model=rf,
        feature_names=FEATURE_COLUMNS,
        model_name="random_forest",
        background_data=X,
    )

    model_imp = pipeline.extract_model_importance()
    assert model_imp is not None
    assert len(model_imp) == 4
    # Árvores ranqueiam petal_length ou petal_width como as mais altas
    assert model_imp[0]["feature"] in ("petal_length", "petal_width")

    report = pipeline.run_pipeline(X, y, n_repeats=3)
    assert len(report.permutation_importance) == 4
    assert report.top_features[0] in ("petal_length", "petal_width")
