"""Testes de Detecção Estatística de Data Drift (PSI e KS-test)."""

import numpy as np
import pandas as pd
import pytest

from app.monitoring.drift import (
    calculate_psi,
    detect_drift,
    drift_report,
    ks_test,
    psi,
    statistical_drift_analysis,
)


def test_ks_test_identical_distributions():
    rng = np.random.default_rng(42)
    sample = rng.normal(loc=0.0, scale=1.0, size=500)

    res = ks_test(sample, sample)
    assert res["statistic"] == pytest.approx(0.0, abs=1e-6)
    assert res["p_value"] == pytest.approx(1.0, abs=1e-3)
    assert res["drift_detected"] is False


def test_ks_test_detects_distribution_drift():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=500)
    shifted = rng.normal(loc=1.5, scale=1.0, size=500)

    res = ks_test(ref, shifted, alpha=0.05)
    assert res["statistic"] > 0.3
    assert res["p_value"] < 0.001
    assert res["drift_detected"] is True


def test_statistical_drift_analysis_stable_and_drifted():
    rng = np.random.default_rng(42)
    df_ref = pd.DataFrame(
        {
            "f1": rng.normal(5.0, 1.0, 500),
            "f2": rng.normal(3.0, 0.5, 500),
        }
    )

    # Dados idênticos ou amostrados da mesma distribuição -> estável
    df_cur_stable = pd.DataFrame(
        {
            "f1": rng.normal(5.0, 1.0, 500),
            "f2": rng.normal(3.0, 0.5, 500),
        }
    )

    analysis_stable = statistical_drift_analysis(df_ref, df_cur_stable)
    assert analysis_stable["status"] == "stable"
    assert len(analysis_stable["drifted_features"]) == 0
    assert analysis_stable["max_psi"] < 0.10

    # Deslocamento severo em f1 -> drift
    df_cur_drift = df_ref.copy()
    df_cur_drift["f1"] = df_cur_drift["f1"] + 5.0

    analysis_drift = statistical_drift_analysis(df_ref, df_cur_drift)
    assert analysis_drift["status"] == "drift"
    assert "f1" in analysis_drift["drifted_features"]
    assert analysis_drift["features"]["f1"]["status"] == "drift"
    assert analysis_drift["features"]["f1"]["ks_drift_detected"] is True


def test_backward_compatibility_drift_report_and_detect_drift():
    rng = np.random.default_rng(20)
    ref = pd.DataFrame({"a": rng.uniform(0, 10, 50), "b": rng.uniform(0, 5, 50)})
    cur = pd.DataFrame({"a": rng.uniform(0, 10, 50), "b": rng.uniform(0, 5, 50)})

    rep = drift_report(ref, cur, features=["a", "b"])
    assert "status" in rep
    assert "features" in rep
    assert "min_ks_p_value" in rep
    assert "a" in rep["features"]

    det = detect_drift(ref, cur, threshold=0.25)
    assert "has_drift" in det
    assert "metrics" in det
    assert "ks_statistic" in det["metrics"]["a"]
