"""Treino, tuning, métricas medidas e explicabilidade."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from app import ml
from app.data import FEATURE_COLUMNS, TARGET_COLUMN, load_iris_dataframe
from app.deployment.gates import evaluate_release_gate
from app.explainability import shapley_explanation
from app.ml import TUNING_SPACES, available_models, build_models, load_artifact, predict_artifact, train_model


def test_every_model_family_fits_and_predicts_probabilities():
    frame = load_iris_dataframe()
    X = frame[FEATURE_COLUMNS]
    y = LabelEncoder().fit_transform(frame[TARGET_COLUMN])
    models = build_models(42)
    assert {"xgboost", "mlp", "svm", "stacking_ensemble", "voting_ensemble"} <= set(models)
    for name, model in models.items():
        probabilities = model.fit(X, y).predict_proba(X)
        assert probabilities.shape == (150, 3), name
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6), name


def test_training_measures_metrics_and_writes_artifact(tmp_path):
    result = train_model(load_iris_dataframe(), "logistic_regression", tmp_path, cv_folds=3)
    metrics = result.metrics
    assert Path(result.artifact_path).parent == tmp_path and Path(result.artifact_path).is_file()
    assert 0.8 <= metrics["accuracy"] <= 1.0
    assert metrics["latency_ms"] > 0, "latência precisa ser medida, não assumida"
    assert 0 <= metrics["brier_score"] <= 2
    assert [item["feature"] for item in metrics["feature_importance"]] != []
    artifact = load_artifact(result.artifact_path)
    assert list(artifact["background"].columns) == FEATURE_COLUMNS
    classes, probabilities = predict_artifact(artifact, [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.8, 2.2]])
    assert classes == ["setosa", "virginica"]
    assert probabilities.shape == (2, 3)


def test_unknown_model_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown model"):
        train_model(load_iris_dataframe(), "nao_existe", tmp_path)


def test_tuning_without_optuna_falls_back_to_randomized_search(tmp_path, monkeypatch):
    """Regressão: `--tune` sem Optuna treinava o modelo padrão sem avisar."""
    monkeypatch.setattr(ml, "optimize_xgboost", lambda *args, **kwargs: None)
    result = train_model(
        load_iris_dataframe(),
        "xgboost",
        tmp_path,
        cv_folds=2,
        tune_hyperparameters=True,
        optimization_trials=2,
    )
    optimization = result.metrics["optimization"]
    assert optimization["method"] == "randomized-search"
    assert optimization["trials"] == 2
    assert set(optimization["best_params"]) == set(TUNING_SPACES["xgboost"])
    tuned = result.model.named_steps["model"].get_params()
    assert tuned["max_depth"] == optimization["best_params"]["model__max_depth"]


def test_tuning_a_model_without_search_space_fails_loudly(tmp_path):
    assert "knn" in available_models()
    with pytest.raises(ValueError, match="no hyperparameter search space"):
        train_model(load_iris_dataframe(), "knn", tmp_path, cv_folds=2, tune_hyperparameters=True)


def test_release_gate_fails_when_latency_was_not_measured():
    """Regressão: sem `latency_ms` o gate passava (ausente era tratado como 0 ms)."""
    gate = evaluate_release_gate({"accuracy": 1.0, "f1_macro": 1.0})
    assert gate["passed"] is False
    latency = next(check for check in gate["checks"] if check["name"] == "latency_ms")
    assert latency == {"name": "latency_ms", "passed": False, "observed": None, "threshold": 100.0}
    assert evaluate_release_gate({"accuracy": 0.95, "f1_macro": 0.95, "latency_ms": 5})["passed"] is True
    assert evaluate_release_gate({"accuracy": 0.85, "f1_macro": 0.95, "latency_ms": 5})["passed"] is False
    assert evaluate_release_gate({"accuracy": 0.95, "f1_macro": 0.95, "latency_ms": 500})["passed"] is False


def test_shapley_values_are_exact_for_a_fitted_model():
    frame = load_iris_dataframe()
    X = frame[FEATURE_COLUMNS]
    y = LabelEncoder().fit_transform(frame[TARGET_COLUMN])
    model = LogisticRegression(max_iter=2000).fit(X, y)
    background = X.sample(40, random_state=0)
    instance = [6.7, 3.0, 5.8, 2.2]
    explanation = shapley_explanation(model, background, instance, class_index=2)
    contributions = {item["feature"]: item["shap_value"] for item in explanation["contributions"]}

    # Eficiência: as contribuições somam exatamente a diferença entre a predição e a base.
    assert sum(contributions.values()) == pytest.approx(
        explanation["prediction"] - explanation["base_value"], abs=1e-6
    )
    assert explanation["prediction"] == pytest.approx(
        model.predict_proba(pd.DataFrame([instance], columns=X.columns))[0, 2]
    )
    assert explanation["base_value"] == pytest.approx(model.predict_proba(background)[:, 2].mean())
    # Uma virginica é reconhecida principalmente pelas pétalas.
    assert explanation["contributions"][0]["feature"] in {"petal_length", "petal_width"}


def test_shapley_gives_zero_to_a_feature_the_model_ignores():
    class OnlyFirstFeature:
        def predict_proba(self, X):
            p = (X.iloc[:, 0] > 0).to_numpy(dtype=float)
            return np.column_stack([1 - p, p])

    background = pd.DataFrame({"a": [-1.0, -1.0, 1.0, -1.0], "b": [0.0, 5.0, -3.0, 9.0]})
    explanation = shapley_explanation(OnlyFirstFeature(), background, [2.0, 100.0], class_index=1)
    values = {item["feature"]: item["shap_value"] for item in explanation["contributions"]}
    assert values["b"] == 0.0
    assert values["a"] == pytest.approx(0.75)
