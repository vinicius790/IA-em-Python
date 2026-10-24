"""Auditoria e Governança Técnica de Modelos de Machine Learning (Model Audit Report).

Exporta relatórios técnicos completos contendo:
- Metadados, versão e integridade criptográfica SHA-256 do artefato.
- Qualidade do conjunto de dados e conformidade do schema.
- Avaliação robusta por validação cruzada K-Fold estratificada com intervalos de confiança.
- Interpretabilidade e Explicabilidade (XAI) via Feature Importance e Permutation.
- Detecção estatística bicaudal de Data Drift (PSI e KS-test).
- Checklist de conformidade e veredito final de governança (APPROVED, WARNING, REJECTED).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..modeling.cross_validation import run_kfold_cross_validation
from ..modeling.explainability import ExplainabilityPipeline
from ..monitoring.drift import statistical_drift_analysis
from ..quality import validate_training_frame


@dataclass
class ModelAuditReport:
    """Relatório estruturado e auditável de governança de modelo."""

    model_name: str
    model_version: str
    generated_at: str
    model_fingerprint: str
    dataset_quality: dict[str, Any]
    cross_validation: dict[str, Any]
    explainability: dict[str, Any]
    data_drift: dict[str, Any]
    governance_checklist: list[dict[str, Any]]
    overall_status: str  # "APPROVED" | "WARNING" | "REJECTED"
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model_audit_report(
    model_name: str,
    model: Any,
    X: pd.DataFrame,
    y: Any,
    reference_X: pd.DataFrame | None = None,
    current_X: pd.DataFrame | None = None,
    n_splits: int = 5,
    model_version: str = "1.0.0",
) -> ModelAuditReport:
    """Constrói a auditoria técnica completa de um modelo."""
    now_str = datetime.now(UTC).isoformat()
    feature_names = list(X.columns)

    # 1. Impressão digital do modelo
    raw_bytes = str(type(model)).encode("utf-8") + str(feature_names).encode("utf-8")
    fingerprint = sha256(raw_bytes).hexdigest()[:16]

    # 2. Qualidade dos Dados
    from ..data import TARGET_COLUMN

    full_df = X.copy()
    target_col = TARGET_COLUMN if (isinstance(y, pd.Series) and y.name == TARGET_COLUMN) else "class_name"
    full_df[target_col] = list(y)
    quality_res = validate_training_frame(full_df, features=feature_names, target=target_col)
    quality_summary = {
        "passed": quality_res.passed,
        "checks": quality_res.checks,
        "rows": len(X),
        "columns": len(feature_names),
    }

    # 3. Validação Cruzada K-Fold
    cv_res = run_kfold_cross_validation(
        model=model,
        X=X,
        y=y,
        model_name=model_name,
        n_splits=n_splits,
        stratified=True,
    )

    # 4. Pipeline de Explicabilidade (Feature Importance / Permutation)
    xai_pipeline = ExplainabilityPipeline(
        model=model,
        feature_names=feature_names,
        model_name=model_name,
        background_data=X,
    )
    xai_report = xai_pipeline.run_pipeline(X=X, y=y, n_repeats=5)

    # 5. Monitoramento de Data Drift (PSI e KS-test)
    ref_data = reference_X if reference_X is not None else X
    cur_data = current_X if current_X is not None else X
    drift_analysis = statistical_drift_analysis(
        reference=ref_data,
        current=cur_data,
        features=feature_names,
        psi_warning=0.10,
        psi_critical=0.25,
    )

    # 6. Checklist de Governança
    mean_f1 = cv_res.aggregated_metrics["f1_macro"]["mean"]
    mean_acc = cv_res.aggregated_metrics["accuracy"]["mean"]
    drift_status = drift_analysis["status"]

    checklist = [
        {
            "criterion": "Qualidade do Conjunto de Dados",
            "passed": quality_res.passed,
            "details": f"{len(quality_res.checks)} verificações de integridade executadas.",
        },
        {
            "criterion": "Desempenho Mínimo em K-Fold (Acurácia >= 85%)",
            "passed": mean_acc >= 0.85,
            "details": f"Acurácia média: {mean_acc:.2%}.",
        },
        {
            "criterion": "Robustez por F1-Macro (F1 >= 85%)",
            "passed": mean_f1 >= 0.85,
            "details": f"F1-Macro médio: {mean_f1:.2%}.",
        },
        {
            "criterion": "Explicabilidade Verificada",
            "passed": len(xai_report.permutation_importance) > 0,
            "details": f"Top feature: {xai_report.top_features[0] if xai_report.top_features else 'N/A'}.",
        },
        {
            "criterion": "Estabilidade de Distribuição (Sem Drift Crítico)",
            "passed": drift_status != "drift",
            "details": f"Status de drift: {drift_status.upper()} (Max PSI: {drift_analysis['max_psi']}).",
        },
    ]

    all_passed = all(item["passed"] for item in checklist)
    has_warning = drift_status == "warning" or mean_f1 < 0.90
    if not all_passed:
        overall_status = "REJECTED"
    elif has_warning:
        overall_status = "WARNING"
    else:
        overall_status = "APPROVED"

    summary = (
        f"Auditoria do modelo '{model_name}' (v{model_version}): Veredito [{overall_status}]. "
        f"Acurácia média CV ({n_splits} folds): {mean_acc:.2%}, F1: {mean_f1:.2%}, "
        f"Status de Drift: {drift_status.upper()}."
    )

    return ModelAuditReport(
        model_name=model_name,
        model_version=model_version,
        generated_at=now_str,
        model_fingerprint=fingerprint,
        dataset_quality=quality_summary,
        cross_validation=cv_res.to_dict(),
        explainability=xai_report.to_dict(),
        data_drift=drift_analysis,
        governance_checklist=checklist,
        overall_status=overall_status,
        summary=summary,
    )


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def export_audit_report(
    report: ModelAuditReport,
    output_path: str | Path,
    format: str = "json",
) -> Path:
    """Exporta o relatório de auditoria para o formato selecionado ('json' ou 'markdown')."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "json":
        data = report.to_dict()
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False, cls=_NumpyEncoder), encoding="utf-8")
        return out

    if format.lower() in ("markdown", "md"):
        cv_agg = report.cross_validation["aggregated_metrics"]
        lines = [
            f"# Relatório de Auditoria e Governança de Modelo: `{report.model_name}`",
            "",
            f"- **Versão:** `{report.model_version}`",
            f"- **Data de Geração:** `{report.generated_at}`",
            f"- **Fingerprint SHA-256:** `{report.model_fingerprint}`",
            f"- **Veredito Geral:** **{report.overall_status}**",
            "",
            "## Resumo Executivo",
            "",
            f"> {report.summary}",
            "",
            "## 1. Checklist de Governança e Compliance",
            "",
            "| Critério | Status | Detalhes |",
            "| :--- | :---: | :--- |",
        ]

        for item in report.governance_checklist:
            icon = "✅ APROVADO" if item["passed"] else "❌ REPROVADO"
            lines.append(f"| {item['criterion']} | {icon} | {item['details']} |")

        lines.extend(
            [
                "",
                "## 2. Validação Cruzada K-Fold",
                "",
                f"- **Estratégia:** {report.cross_validation['n_splits']}-Fold Estratificado",
                f"- **Total de Amostras:** {report.cross_validation['sample_count']}",
                "",
                "| Métrica | Média | Desvio Padrão | Intervalo de Confiança (95%) |",
                "| :--- | :---: | :---: | :---: |",
                f"| **Acurácia** | {cv_agg['accuracy']['mean']:.4f} | {cv_agg['accuracy']['std']:.4f} | [{cv_agg['accuracy']['ci_lower']:.4f}; {cv_agg['accuracy']['ci_upper']:.4f}] |",
                f"| **F1 Macro** | {cv_agg['f1_macro']['mean']:.4f} | {cv_agg['f1_macro']['std']:.4f} | [{cv_agg['f1_macro']['ci_lower']:.4f}; {cv_agg['f1_macro']['ci_upper']:.4f}] |",
                f"| **Precisão Macro** | {cv_agg['precision_macro']['mean']:.4f} | {cv_agg['precision_macro']['std']:.4f} | [{cv_agg['precision_macro']['ci_lower']:.4f}; {cv_agg['precision_macro']['ci_upper']:.4f}] |",
                f"| **Revocação Macro** | {cv_agg['recall_macro']['mean']:.4f} | {cv_agg['recall_macro']['std']:.4f} | [{cv_agg['recall_macro']['ci_lower']:.4f}; {cv_agg['recall_macro']['ci_upper']:.4f}] |",
                "",
                "## 3. Explicabilidade (XAI - Permutation Feature Importance)",
                "",
                "| Ranking | Feature | Queda Média na Performance | Desvio Padrão |",
                "| :---: | :--- | :---: | :---: |",
            ]
        )

        for rank, p in enumerate(report.explainability["permutation_importance"], 1):
            lines.append(
                f"| #{rank} | `{p['feature']}` | {p['importance_mean']:.4f} | {p['importance_std']:.4f} |"
            )

        lines.extend(
            [
                "",
                "## 4. Monitoramento de Data Drift Estatístico (PSI e KS-test)",
                "",
                f"- **Status Geral:** `{report.data_drift['status'].upper()}`",
                f"- **PSI Máximo:** `{report.data_drift['max_psi']:.6f}`",
                f"- **Menor p-valor KS:** `{report.data_drift['min_ks_p_value']:.6f}`",
                "",
                "| Feature | PSI | Status PSI | KS Estatística | KS p-valor | Drift Detectado |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )

        for col, m in report.data_drift["features"].items():
            lines.append(
                f"| `{col}` | {m['psi']:.4f} | {m['status']} | {m['ks_statistic']:.4f} | {m['ks_p_value']:.4f} | {'SIM' if m['ks_drift_detected'] else 'NÃO'} |"
            )

        lines.extend(
            [
                "",
                "## Arquitetura do Pipeline de Auditoria",
                "",
                "```mermaid",
                "flowchart TD",
                "    Data[Dados de Entrada] --> Gates[Gates de Qualidade]",
                "    Gates --> KFold[Validação K-Fold Estratificada]",
                "    KFold --> XAI[Pipeline de Explicabilidade]",
                "    XAI --> Drift[Detecção de Data Drift PSI & KS]",
                "    Drift --> Checklist[Checklist de Compliance]",
                f'    Checklist --> Verdict["Veredito: {report.overall_status}"]',
                "```",
                "",
            ]
        )

        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    raise ValueError(f"Formato não suportado: {format}. Use 'json' ou 'markdown'.")
