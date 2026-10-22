"""Testes automatizados do Módulo de IA Ética, Justiça e Auditoria de Viés (Fairness & Bias Auditor)."""

import numpy as np
import pytest

from app.governance.fairness import (
    FairnessAuditReport,
    FairnessMetrics,
    audit_fairness,
)


def test_fairness_audit_perfect_parity():
    # Cenário perfeitamente equilibrado: ambos os grupos com as mesmas taxas
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0, 1, 0]
    prot = ["A", "A", "A", "A", "B", "B", "B", "B"]

    report = audit_fairness(
        y_true=y_true,
        y_pred=y_pred,
        protected_attribute=prot,
        positive_label=1,
        attribute_name="gender",
    )

    assert isinstance(report, FairnessAuditReport)
    assert report.protected_attribute == "gender"
    assert report.total_samples == 8
    assert report.fairness_metrics.disparate_impact_ratio == pytest.approx(1.0, abs=1e-4)
    assert report.fairness_metrics.demographic_parity_difference == pytest.approx(0.0, abs=1e-4)
    assert report.fairness_metrics.equal_opportunity_difference == pytest.approx(0.0, abs=1e-4)
    assert report.fairness_metrics.average_odds_difference == pytest.approx(0.0, abs=1e-4)
    assert report.fairness_metrics.is_disparate_impact_compliant is True
    assert report.fairness_metrics.is_equal_opportunity_compliant is True
    assert report.status == "compliant"
    assert len(report.recommendations) >= 1
    assert report.recommendations[0].stage == "monitoring"


def test_fairness_audit_detects_disparate_impact_violation():
    # Grupo A (Privilegiado): 8 aprovações em 10 (selection_rate = 0.8)
    # Grupo B (Desprivilegiado): 2 aprovações em 10 (selection_rate = 0.2)
    # Disparate Impact = 0.2 / 0.8 = 0.25 (viola fortemente o limite de 0.80)
    y_true = [1] * 8 + [0] * 2 + [1] * 4 + [0] * 6
    y_pred = [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8
    prot = ["A"] * 10 + ["B"] * 10

    report = audit_fairness(
        y_true=y_true,
        y_pred=y_pred,
        protected_attribute=prot,
        privileged_group="A",
        attribute_name="ethnicity",
    )

    metrics = report.fairness_metrics
    assert metrics.disparate_impact_ratio == pytest.approx(0.25, abs=1e-2)
    assert metrics.is_disparate_impact_compliant is False
    assert report.status in ("biased", "warning")

    # Verifica se gerou recomendação de pré-processamento (Reweighting)
    stages = [r.stage for r in report.recommendations]
    assert "preprocessing" in stages
    assert any("Reweighting" in r.title for r in report.recommendations)


def test_fairness_audit_equal_opportunity_difference():
    # TPR do grupo A é 100%, TPR do grupo B é 33%
    y_true = [1, 1, 1, 0] + [1, 1, 1, 0]
    y_pred = [1, 1, 1, 0] + [1, 0, 0, 0]
    prot = ["privileged"] * 4 + ["unprivileged"] * 4

    report = audit_fairness(
        y_true=y_true,
        y_pred=y_pred,
        protected_attribute=prot,
        privileged_group="privileged",
        equal_opportunity_threshold=0.1,
    )

    metrics = report.fairness_metrics
    assert metrics.equal_opportunity_difference > 0.5
    assert metrics.is_equal_opportunity_compliant is False

    # Deve sugerir calibração de limiares por grupo no pós-processamento
    post_recs = [r for r in report.recommendations if r.stage == "postprocessing"]
    assert len(post_recs) >= 1
    assert "Limiares" in post_recs[0].title or "Equalized" in post_recs[0].title


def test_fairness_audit_multiclass_groups_and_serialization():
    # Três subgrupos de idade: Jovem, Adulto, Idoso
    y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 0, 1, 0, 0, 1, 0, 1]
    prot = ["Jovem", "Jovem", "Jovem", "Adulto", "Adulto", "Adulto", "Idoso", "Idoso", "Idoso"]

    report = audit_fairness(y_true=y_true, y_pred=y_pred, protected_attribute=prot, attribute_name="age_group")
    d = report.to_dict()

    assert "protected_attribute" in d
    assert "fairness_metrics" in d
    assert "group_metrics" in d
    assert set(d["group_metrics"].keys()) == {"Jovem", "Adulto", "Idoso"}
    assert "recommendations" in d


def test_fairness_audit_validation_errors():
    with pytest.raises(ValueError, match="mesma quantidade"):
        audit_fairness(y_true=[1, 0], y_pred=[1], protected_attribute=["A", "B"])

    with pytest.raises(ValueError, match="pelo menos 2 grupos"):
        audit_fairness(y_true=[1, 0], y_pred=[1, 0], protected_attribute=["A", "A"])
