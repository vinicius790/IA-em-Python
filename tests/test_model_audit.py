"""Testes do Módulo de Auditoria e Governança de Modelos (Model Audit Report)."""

import json
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

from app.data import FEATURE_COLUMNS, TARGET_COLUMN, load_iris_dataframe
from app.governance.model_audit import (
    ModelAuditReport,
    build_model_audit_report,
    export_audit_report,
)


@pytest.fixture(scope="module")
def trained_logistic_model():
    df = load_iris_dataframe()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_build_and_export_model_audit_report(trained_logistic_model, tmp_path):
    model, X, y = trained_logistic_model

    report = build_model_audit_report(
        model_name="logistic_regression",
        model=model,
        X=X,
        y=y,
        n_splits=3,
        model_version="1.0.0",
    )

    assert isinstance(report, ModelAuditReport)
    assert report.model_name == "logistic_regression"
    assert report.model_version == "1.0.0"
    assert report.overall_status in ("APPROVED", "WARNING", "REJECTED")
    assert len(report.governance_checklist) == 5
    assert report.dataset_quality["passed"] is True
    assert "aggregated_metrics" in report.cross_validation
    assert "permutation_importance" in report.explainability
    assert "features" in report.data_drift

    # Teste de exportação para JSON
    json_path = tmp_path / "audit_report.json"
    res_json = export_audit_report(report, json_path, format="json")
    assert res_json.exists()
    loaded_json = json.loads(res_json.read_text(encoding="utf-8"))
    assert loaded_json["model_name"] == "logistic_regression"
    assert "cross_validation" in loaded_json

    # Teste de exportação para Markdown
    md_path = tmp_path / "audit_report.md"
    res_md = export_audit_report(report, md_path, format="markdown")
    assert res_md.exists()
    content = res_md.read_text(encoding="utf-8")
    assert "# Relatório de Auditoria e Governança de Modelo: `logistic_regression`" in content
    assert "```mermaid" in content
    assert "Validação Cruzada K-Fold" in content
    assert "Explicabilidade (XAI" in content
    assert "Monitoramento de Data Drift" in content
