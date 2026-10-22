"""Detecção de drift estatístico de dados (PSI - Population Stability Index e KS-test).

Fornece análise unificada de Data Drift bicaudal e distribuição multivariada:
- Population Stability Index (PSI): quantifica mudança na distribuição discretizada em bins/quantis.
- Teste Kolmogorov-Smirnov (KS-test): teste não paramétrico contínuo de igualdade de distribuições 1D.
- Classificação de severidade: 'stable' (<0.10), 'warning' (0.10-0.25), 'drift' (>=0.25 ou p < 0.01).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def psi(
    reference: Union[Sequence[float], np.ndarray],
    current: Union[Sequence[float], np.ndarray],
    bins: int = 10,
) -> float:
    """Calcula o Population Stability Index (PSI) entre a distribuição de referência e a atual."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(ref, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Trata valores idênticos nas bordas dos percentis
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    ref_counts = np.histogram(ref, bins=bin_edges)[0]
    cur_counts = np.histogram(cur, bins=bin_edges)[0]

    ref_pct = np.where(ref_counts == 0, 0.0001, ref_counts) / len(ref)
    cur_pct = np.where(cur_counts == 0, 0.0001, cur_counts) / len(cur)

    val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(val)


calculate_psi = psi


def ks_test(
    reference: Union[Sequence[float], np.ndarray],
    current: Union[Sequence[float], np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Calcula o teste Kolmogorov-Smirnov (KS-test) bicaudal para duas distribuições.

    Retorna:
    - statistic: valor máximo D entre as CDFs empíricas das duas amostras.
    - p_value: significância estatística da diferença observada.
    - drift_detected: booleano indicando se p_value < alpha.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    n1 = len(ref)
    n2 = len(cur)

    if n1 == 0 or n2 == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "alpha": alpha,
            "sample_sizes": (n1, n2),
        }

    try:
        from scipy import stats

        res = stats.ks_2samp(ref, cur)
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        # Fallback analítico com NumPy puro
        all_vals = np.sort(np.unique(np.concatenate([ref, cur])))
        cdf1 = np.searchsorted(np.sort(ref), all_vals, side="right") / n1
        cdf2 = np.searchsorted(np.sort(cur), all_vals, side="right") / n2
        stat = float(np.max(np.abs(cdf1 - cdf2)))

        # Aproximação assintótica de Kolmogorov
        en = np.sqrt((n1 * n2) / (n1 + n2))
        lambda_val = (en + 0.12 + 0.11 / en) * stat
        if lambda_val <= 0:
            p_val = 1.0
        else:
            p_val = 0.0
            for k in range(1, 101):
                term = 2.0 * ((-1) ** (k - 1)) * np.exp(-2.0 * (k * lambda_val) ** 2)
                p_val += term
                if abs(term) < 1e-10:
                    break
            p_val = float(np.clip(p_val, 0.0, 1.0))

    drift_detected = bool(p_val < alpha)
    return {
        "statistic": round(stat, 6),
        "p_value": round(p_val, 6),
        "drift_detected": drift_detected,
        "alpha": alpha,
        "sample_sizes": (n1, n2),
    }


def statistical_drift_analysis(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Optional[List[str]] = None,
    psi_warning: float = 0.10,
    psi_critical: float = 0.25,
    ks_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Análise estatística completa de Data Drift combinando PSI e KS-test em cada feature."""
    cols = features or list(reference.select_dtypes(include=[np.number]).columns)

    if len(current) < 5:
        return {
            "status": "insufficient_data",
            "required": 5,
            "samples": len(current),
            "features": {},
            "drifted_features": [],
            "max_psi": 0.0,
            "min_ks_p_value": 1.0,
        }

    feature_metrics: Dict[str, Any] = {}
    drifted_features: List[str] = []
    max_psi = 0.0
    min_ks_p = 1.0

    for col in cols:
        if col in reference.columns and col in current.columns:
            ref_vals = reference[col].dropna().values
            cur_vals = current[col].dropna().values

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            psi_val = psi(ref_vals, cur_vals)
            ks_res = ks_test(ref_vals, cur_vals, alpha=ks_alpha)

            # Classificação integrada:
            # - drift: PSI >= critical OU (PSI >= warning e KS detecta p < 0.01)
            # - warning: PSI >= warning OU KS detecta p < alpha
            # - stable: caso contrário
            if psi_val >= psi_critical or (psi_val >= psi_warning and ks_res["p_value"] < 0.01):
                col_status = "drift"
            elif psi_val >= psi_warning or ks_res["drift_detected"]:
                col_status = "warning"
            else:
                col_status = "stable"

            feature_metrics[col] = {
                "psi": round(psi_val, 6),
                "ks_statistic": ks_res["statistic"],
                "ks_p_value": ks_res["p_value"],
                "ks_drift_detected": ks_res["drift_detected"],
                "status": col_status,
            }

            if col_status == "drift":
                drifted_features.append(col)

            if psi_val > max_psi:
                max_psi = psi_val
            if ks_res["p_value"] < min_ks_p:
                min_ks_p = ks_res["p_value"]

    if drifted_features:
        overall = "drift"
    elif any(f["status"] == "warning" for f in feature_metrics.values()):
        overall = "warning"
    else:
        overall = "stable"

    return {
        "status": overall,
        "features": feature_metrics,
        "drifted_features": drifted_features,
        "max_psi": round(max_psi, 6),
        "min_ks_p_value": round(min_ks_p, 6),
        "warning_threshold": psi_warning,
        "critical_threshold": psi_critical,
        "ks_alpha": ks_alpha,
        "reference_rows": len(reference),
        "current_rows": len(current),
    }


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: List[str],
    warning: float = 0.10,
    critical: float = 0.25,
) -> Dict[str, Any]:
    """Relatório estruturado de drift por coluna com classificação de estabilidade e KS-test."""
    if len(current) < 30:
        return {
            "status": "insufficient_data",
            "required": 30,
            "samples": len(current),
            "features": {},
            "drifted_features": [],
            "max_psi": 0.0,
        }

    analysis = statistical_drift_analysis(
        reference=reference,
        current=current,
        features=features,
        psi_warning=warning,
        psi_critical=critical,
    )

    # Mantém compatibilidade com os consumidores da assinatura original
    return {
        "status": analysis["status"],
        "features": analysis["features"],
        "drifted_features": analysis["drifted_features"],
        "max_psi": analysis["max_psi"],
        "min_ks_p_value": analysis.get("min_ks_p_value", 1.0),
        "warning_threshold": warning,
        "critical_threshold": critical,
    }


def detect_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, threshold: float = 0.25) -> Dict[str, Any]:
    """Detecta Data Drift usando PSI e KS-test para colunas numéricas (compatibilidade)."""
    analysis = statistical_drift_analysis(
        reference=reference_data,
        current=current_data,
        psi_critical=threshold,
    )

    drift_results = {}
    for col, data in analysis["features"].items():
        drift_results[col] = {
            "psi": data["psi"],
            "ks_statistic": data["ks_statistic"],
            "ks_p_value": data["ks_p_value"],
            "drift": data["status"] == "drift",
        }

    return {"has_drift": analysis["status"] == "drift", "metrics": drift_results, "analysis": analysis}
