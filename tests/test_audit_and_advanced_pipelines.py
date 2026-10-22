"""Testes completos dos pipelines avançados de IA:
- Validação cruzada K-Fold estratificada com intervalos de confiança
- Detecção estatística de Data Drift (PSI e KS-test)
- Pipeline de Explicabilidade (Feature Importance nativa e Permutation)
- Auditoria e Governança Técnica de Modelos com exportação (Markdown e JSON)
- Comandos avançados da CLI Typer
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from typer.testing import CliRunner

from app.cli import app as cli_app
from app.data import FEATURE_COLUMNS, TARGET_COLUMN, load_iris_dataframe
from app.governance.model_audit import (
    ModelAuditReport,
    build_model_audit_report,
    export_audit_report,
)
from app.modeling.cross_validation import KFoldValidationResult, run_kfold_cross_validation
from app.modeling.explainability import (
    ExplainabilityPipeline,
    ExplainabilityReport,
    LocalPredictionDecomposition,
    compute_kernel_shap,
    pure_permutation_importance,
)
from app.monitoring.drift import (
    detect_drift,
    drift_report,
    ks_test,
    psi,
    statistical_drift_analysis,
)

runner = CliRunner()


@pytest.fixture(scope="module")
def iris_data():
    df = load_iris_dataframe()
    X = df[FEATURE_COLUMNS]
    y_raw = df[TARGET_COLUMN]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    return df, X, y, encoder


# -----------------------------------------------------------------------------
# 1. Testes de Validação Cruzada K-Fold
# -----------------------------------------------------------------------------
def test_kfold_cross_validation_stratified(iris_data):
    _, X, y, _ = iris_data
    model = LogisticRegression(max_iter=500, random_state=42)

    res = run_kfold_cross_validation(
        model=model,
        X=X,
        y=y,
        model_name="logistic_regression",
        n_splits=5,
        stratified=True,
    )

    assert isinstance(res, KFoldValidationResult)
    assert res.n_splits == 5
    assert res.sample_count == len(X)
    assert len(res.fold_metrics) == 5

    # Métricas agregadas e intervalos de confiança
    for metric in ("accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"):
        assert metric in res.aggregated_metrics
        m = res.aggregated_metrics[metric]
        assert 0.0 <= m["mean"] <= 1.0
        assert 0.0 <= m["std"] <= 1.0
        assert m["ci_lower"] <= m["mean"] <= m["ci_upper"]

    # Matriz de confusão out-of-fold
    cm = np.array(res.confusion_matrix)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(X)

    # Serialização
    res_dict = res.to_dict()
    assert res_dict["model_name"] == "logistic_regression"
    assert "aggregated_metrics" in res_dict


def test_kfold_cross_validation_unstratified_and_validation_errors(iris_data):
    _, X, y, _ = iris_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    res = run_kfold_cross_validation(
        model=model,
        X=X.to_numpy(),
        y=y,
        model_name="random_forest",
        n_splits=3,
        stratified=False,
    )
    assert res.n_splits == 3
    assert len(res.fold_metrics) == 3

    with pytest.raises(ValueError, match="n_splits deve ser pelo menos 2"):
        run_kfold_cross_validation(model, X, y, n_splits=1)

    with pytest.raises(ValueError, match="não pode ser maior que o número de amostras"):
        run_kfold_cross_validation(model, X.head(4), y[:4], n_splits=5)


# -----------------------------------------------------------------------------
# 2. Testes de Detecção Estatística de Data Drift (PSI e KS-test)
# -----------------------------------------------------------------------------
def test_psi_and_ks_test_statistical_properties():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=1000)
    cur_same = rng.normal(loc=0.0, scale=1.0, size=1000)
    cur_drift = rng.normal(loc=2.5, scale=1.0, size=1000)

    # Teste de PSI
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-5)
    assert psi(ref, cur_same) < 0.10
    assert psi(ref, cur_drift) > 0.50

    # Teste de Kolmogorov-Smirnov (KS-test)
    ks_same = ks_test(ref, cur_same, alpha=0.05)
    assert not ks_same["drift_detected"]
    assert ks_same["p_value"] > 0.01

    ks_drift = ks_test(ref, cur_drift, alpha=0.05)
    assert ks_drift["drift_detected"]
    assert ks_drift["p_value"] < 0.001
    assert ks_drift["statistic"] > 0.30

    # Casos de borda
    empty_ks = ks_test([], [1.0, 2.0])
    assert empty_ks["statistic"] == 0.0
    assert empty_ks["p_value"] == 1.0


def test_statistical_drift_analysis_multivariate(iris_data):
    _, X, _, _ = iris_data
    ref = X.copy()
    cur_drifted = X.copy()
    # Injetar drift forte em petal_length
    cur_drifted["petal_length"] = cur_drifted["petal_length"] + 2.0

    analysis = statistical_drift_analysis(
        reference=ref,
        current=cur_drifted,
        features=FEATURE_COLUMNS,
        psi_warning=0.10,
        psi_critical=0.25,
        ks_alpha=0.05,
    )

    assert analysis["status"] == "drift"
    assert "petal_length" in analysis["drifted_features"]
    assert analysis["features"]["petal_length"]["status"] == "drift"
    assert analysis["features"]["petal_length"]["ks_drift_detected"] is True
    assert analysis["max_psi"] > 0.25

    # Dados insuficientes
    small_analysis = statistical_drift_analysis(reference=ref, current=X.head(3))
    assert small_analysis["status"] == "insufficient_data"

    # Compatibilidade com drift_report e detect_drift
    rep = drift_report(ref, cur_drifted, FEATURE_COLUMNS)
    assert rep["status"] == "drift"

    det = detect_drift(ref, cur_drifted)
    assert det["has_drift"] is True


# -----------------------------------------------------------------------------
# 3. Testes do Pipeline de Explicabilidade (XAI)
# -----------------------------------------------------------------------------
def test_explainability_pipeline_and_waterfall(iris_data):
    _, X, y, _ = iris_data
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)

    pipeline = ExplainabilityPipeline(
        model=model,
        feature_names=FEATURE_COLUMNS,
        model_name="logistic_regression",
        background_data=X,
    )

    # Feature Importance nativa do modelo
    model_imp = pipeline.extract_model_importance()
    assert model_imp is not None
    assert len(model_imp) == 4
    for item in model_imp:
        assert "feature" in item
        assert "importance" in item

    # Permutation Feature Importance pura
    perm_imp = pipeline.run_permutation_importance(X, y, n_repeats=3, seed=42)
    assert len(perm_imp) == 4
    for i in range(len(perm_imp) - 1):
        assert perm_imp[i]["importance_mean"] >= perm_imp[i + 1]["importance_mean"]

    # Explicabilidade local de instância (Kernel Shapley)
    sample_instance = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    local_decomp = pipeline.explain_instance(sample_instance, target_class=0)

    assert isinstance(local_decomp, LocalPredictionDecomposition)
    total_contrib = sum(c.contribution for c in local_decomp.contributions)
    assert local_decomp.base_value + total_contrib == pytest.approx(local_decomp.prediction, abs=1e-3)
    assert len(local_decomp.waterfall_steps) == 6  # base + 4 features + final

    # Execução do pipeline completo consolidado
    report = pipeline.run_pipeline(X, y, sample_instances=[sample_instance], n_repeats=3)
    assert isinstance(report, ExplainabilityReport)
    assert report.model_name == "logistic_regression"
    assert len(report.sample_explanations) == 1
    assert len(report.top_features) > 0


def test_explainability_pipeline_tree_models(iris_data):
    _, X, y, _ = iris_data
    tree = RandomForestClassifier(n_estimators=10, random_state=42)
    tree.fit(X, y)

    pipeline = ExplainabilityPipeline(
        model=tree,
        feature_names=FEATURE_COLUMNS,
        model_name="random_forest",
        background_data=X,
    )
    tree_imp = pipeline.extract_model_importance()
    assert tree_imp is not None
    assert len(tree_imp) == 4
    assert sum(item["relative_percentage"] for item in tree_imp) == pytest.approx(100.0, abs=1.0)


# -----------------------------------------------------------------------------
# 4. Testes do Relatório de Auditoria e Governança do Modelo
# -----------------------------------------------------------------------------
def test_model_audit_report_generation_and_export(iris_data, tmp_path):
    _, X, y, _ = iris_data
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)

    report = build_model_audit_report(
        model_name="logistic_regression",
        model=model,
        X=X,
        y=y,
        n_splits=3,
        model_version="2.0.0",
    )

    assert isinstance(report, ModelAuditReport)
    assert report.overall_status in ("APPROVED", "WARNING", "REJECTED")
    assert len(report.governance_checklist) == 5
    assert len(report.model_fingerprint) == 16
    assert report.dataset_quality["passed"] is True

    # Exportação em Markdown com diagrama Mermaid
    md_path = tmp_path / "audit_report.md"
    exported_md = export_audit_report(report, md_path, format="markdown")
    assert exported_md.exists()
    content = exported_md.read_text(encoding="utf-8")
    assert "# Relatório de Auditoria e Governança de Modelo" in content
    assert "```mermaid" in content
    assert "flowchart TD" in content
    assert "Acurácia" in content

    # Exportação em JSON
    json_path = tmp_path / "audit_report.json"
    exported_json = export_audit_report(report, json_path, format="json")
    assert exported_json.exists()
    parsed_json = json.loads(exported_json.read_text(encoding="utf-8"))
    assert parsed_json["model_name"] == "logistic_regression"
    assert "governance_checklist" in parsed_json

    # Formato inválido
    with pytest.raises(ValueError, match="Formato não suportado"):
        export_audit_report(report, tmp_path / "audit.xml", format="xml")


# -----------------------------------------------------------------------------
# 5. Testes dos Comandos Avançados da CLI
# -----------------------------------------------------------------------------
def test_cli_explain_pipeline():
    result = runner.invoke(cli_app, ["explain-pipeline", "--model", "logistic_regression", "--repeats", "2"])
    assert result.exit_code == 0, result.output
    assert "Ranking" in result.output
    assert "auditado via XAI" in result.output


def test_cli_kfold():
    result = runner.invoke(cli_app, ["kfold", "--model", "logistic_regression", "--folds", "3"])
    assert result.exit_code == 0, result.output
    assert "Acurácia" in result.output or "accuracy" in result.output
    assert "IC 95%" in result.output


def test_cli_drift_check():
    result = runner.invoke(cli_app, ["drift-check", "--shift", "1.5"])
    assert result.exit_code == 0, result.output
    assert "PSI" in result.output
    assert "KS" in result.output
    assert "Status Geral de Drift" in result.output


def test_cli_audit_export(tmp_path):
    md_out = tmp_path / "cli_audit.md"
    result_md = runner.invoke(
        cli_app, ["audit-export", "--model", "logistic_regression", "--output", str(md_out), "--format", "markdown"]
    )
    assert result_md.exit_code == 0, result_md.output
    assert md_out.exists()
    assert "Veredito de Governança" in result_md.output

    json_out = tmp_path / "cli_audit.json"
    result_json = runner.invoke(
        cli_app, ["audit-export", "--model", "logistic_regression", "--output", str(json_out), "--format", "json"]
    )
    assert result_json.exit_code == 0, result_json.output
    assert json_out.exists()
