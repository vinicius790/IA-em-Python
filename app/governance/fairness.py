"""Módulo de IA Ética, Justiça e Auditoria de Viés (Fairness & Bias Auditor).

Implementa:
- Métricas de Paridade Demográfica (Demographic Parity) e Impacto Disparate (Disparate Impact).
- Igualdade de Oportunidade (Equal Opportunity / True Positive Rate Parity).
- Odds Balanceadas (Equalized Odds / Average Odds Difference).
- Auditoria de atributos protegidos com relatórios acionáveis e recomendações de mitigação.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GroupConfusionMetrics:
    """Métricas de confusão e taxas de classificação de um grupo específico."""

    group_value: Any
    sample_size: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    selection_rate: float  # Taxa de predição positiva P(Y_hat = 1)
    base_rate: float  # Taxa de positivos reais P(Y = 1)
    true_positive_rate: float  # Recall / Sensibilidade TPR
    false_positive_rate: float  # FPR
    accuracy: float
    precision: float


@dataclass
class FairnessMetrics:
    """Consolidado de métricas de justiça ética e impacto algorítmico."""

    privileged_group: Any
    unprivileged_group: Any
    # Demographic Parity
    disparate_impact_ratio: float  # P(Y_hat=1|unprivileged) / P(Y_hat=1|privileged)
    demographic_parity_difference: float  # |P(Y_hat=1|unprivileged) - P(Y_hat=1|privileged)|
    # Equal Opportunity
    equal_opportunity_difference: float  # |TPR_unprivileged - TPR_privileged|
    # Equalized Odds
    false_positive_rate_difference: float  # |FPR_unprivileged - FPR_privileged|
    average_odds_difference: float  # 0.5 * (|FPR_diff| + |TPR_diff|)
    # Status de conformidade
    is_disparate_impact_compliant: bool  # Regra dos 80% (0.80 <= ratio <= 1.25)
    is_equal_opportunity_compliant: bool  # diff <= 0.10
    overall_status: str  # "compliant", "warning", "biased"


@dataclass
class MitigationRecommendation:
    """Recomendação acionável para redução de viés algorítmico."""

    stage: str  # "preprocessing", "inprocessing", "postprocessing"
    title: str
    description: str
    action_item: str


@dataclass
class FairnessAuditReport:
    """Relatório final de auditoria de justiça algorítmica."""

    protected_attribute: str
    total_samples: int
    group_metrics: dict[str, dict[str, Any]]
    fairness_metrics: FairnessMetrics
    status: str
    recommendations: list[MitigationRecommendation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protected_attribute": self.protected_attribute,
            "total_samples": self.total_samples,
            "group_metrics": self.group_metrics,
            "fairness_metrics": asdict(self.fairness_metrics),
            "status": self.status,
            "recommendations": [asdict(r) for r in self.recommendations],
        }


def _compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_value: Any,
    positive_label: Any = 1,
) -> GroupConfusionMetrics:
    """Calcula taxas e matriz de confusão para um subgrupo."""
    n = len(y_true)
    if n == 0:
        return GroupConfusionMetrics(
            group_value=group_value,
            sample_size=0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            selection_rate=0.0,
            base_rate=0.0,
            true_positive_rate=0.0,
            false_positive_rate=0.0,
            accuracy=0.0,
            precision=0.0,
        )

    tp = int(np.sum((y_pred == positive_label) & (y_true == positive_label)))
    fp = int(np.sum((y_pred == positive_label) & (y_true != positive_label)))
    tn = int(np.sum((y_pred != positive_label) & (y_true != positive_label)))
    fn = int(np.sum((y_pred != positive_label) & (y_true == positive_label)))

    selection_rate = float((tp + fp) / n)
    base_rate = float((tp + fn) / n)
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    accuracy = float((tp + tn) / n)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return GroupConfusionMetrics(
        group_value=group_value,
        sample_size=n,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        selection_rate=round(selection_rate, 4),
        base_rate=round(base_rate, 4),
        true_positive_rate=round(tpr, 4),
        false_positive_rate=round(fpr, 4),
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
    )


def audit_fairness(
    y_true: Sequence[Any] | np.ndarray | pd.Series,
    y_pred: Sequence[Any] | np.ndarray | pd.Series,
    protected_attribute: Sequence[Any] | np.ndarray | pd.Series,
    positive_label: Any = 1,
    privileged_group: Any | None = None,
    disparate_impact_threshold: float = 0.8,
    equal_opportunity_threshold: float = 0.1,
    attribute_name: str = "protected_attribute",
) -> FairnessAuditReport:
    """Executa auditoria ética abrangente de viés e justiça algorítmica.

    Calcula métricas fundamentais da literatura de IA Responsável (Fairness):
    1. Disparate Impact (Regra dos 4/5 - EEOC):
       Ratio = P(Y_hat=1 | Unprivileged) / P(Y_hat=1 | Privileged)
    2. Demographic Parity Difference:
       |P(Y_hat=1 | Unprivileged) - P(Y_hat=1 | Privileged)|
    3. Equal Opportunity Difference:
       |TPR_unprivileged - TPR_privileged|
    4. Average Odds Difference:
       0.5 * (|FPR_diff| + |TPR_diff|)
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    prot = np.asarray(protected_attribute)

    if not (len(yt) == len(yp) == len(prot)):
        raise ValueError("y_true, y_pred e protected_attribute devem possuir a mesma quantidade de elementos")

    unique_groups = list(pd.unique(prot))
    if len(unique_groups) < 2:
        raise ValueError(
            f"O atributo protegido deve conter pelo menos 2 grupos distintos. Encontrados: {unique_groups}"
        )

    # Definir grupo privilegiado (se não fornecido, adota o grupo com maior selection rate ou o primeiro com maior n)
    group_subsets = {g: (yt[prot == g], yp[prot == g]) for g in unique_groups}
    group_metrics_map: dict[str, GroupConfusionMetrics] = {}

    for g, (g_yt, g_yp) in group_subsets.items():
        group_metrics_map[str(g)] = _compute_group_metrics(
            g_yt, g_yp, group_value=g, positive_label=positive_label
        )

    if privileged_group is None or str(privileged_group) not in group_metrics_map:
        # Padrão: grupo com maior taxa de aprovação/seleção (selection_rate)
        sorted_groups = sorted(
            group_metrics_map.keys(),
            key=lambda k: (group_metrics_map[k].selection_rate, group_metrics_map[k].sample_size),
            reverse=True,
        )
        priv_key = sorted_groups[0]
        unpriv_key = sorted_groups[-1]
    else:
        priv_key = str(privileged_group)
        other_keys = [k for k in group_metrics_map if k != priv_key]
        # Pega o grupo não privilegiado com menor selection_rate entre os demais
        unpriv_key = min(other_keys, key=lambda k: group_metrics_map[k].selection_rate)

    priv_metrics = group_metrics_map[priv_key]
    unpriv_metrics = group_metrics_map[unpriv_key]

    # 1. Disparate Impact
    if priv_metrics.selection_rate > 0:
        di_ratio = unpriv_metrics.selection_rate / priv_metrics.selection_rate
    else:
        di_ratio = 1.0 if unpriv_metrics.selection_rate == 0 else float("inf")

    # 2. Demographic Parity Difference
    dp_diff = abs(unpriv_metrics.selection_rate - priv_metrics.selection_rate)

    # 3. Equal Opportunity Difference
    eq_opp_diff = abs(unpriv_metrics.true_positive_rate - priv_metrics.true_positive_rate)

    # 4. Odds Balanceadas
    fpr_diff = abs(unpriv_metrics.false_positive_rate - priv_metrics.false_positive_rate)
    avg_odds_diff = 0.5 * (fpr_diff + eq_opp_diff)

    # Avaliação de conformidade
    is_di_compliant = bool(disparate_impact_threshold <= di_ratio <= (1.0 / disparate_impact_threshold))
    is_eq_opp_compliant = bool(eq_opp_diff <= equal_opportunity_threshold)

    if is_di_compliant and is_eq_opp_compliant:
        overall_status = "compliant"
    elif not is_di_compliant and not is_eq_opp_compliant:
        overall_status = "biased"
    else:
        overall_status = "warning"

    fairness_metrics = FairnessMetrics(
        privileged_group=priv_key,
        unprivileged_group=unpriv_key,
        disparate_impact_ratio=round(float(di_ratio), 4),
        demographic_parity_difference=round(float(dp_diff), 4),
        equal_opportunity_difference=round(float(eq_opp_diff), 4),
        false_positive_rate_difference=round(float(fpr_diff), 4),
        average_odds_difference=round(float(avg_odds_diff), 4),
        is_disparate_impact_compliant=is_di_compliant,
        is_equal_opportunity_compliant=is_eq_opp_compliant,
        overall_status=overall_status,
    )

    # Gerar recomendações caso haja disparidade
    recommendations: list[MitigationRecommendation] = []
    if overall_status != "compliant":
        if not is_di_compliant:
            recommendations.append(
                MitigationRecommendation(
                    stage="preprocessing",
                    title="Reponderação de Amostras (Reweighting)",
                    description=(
                        f"O Disparate Impact Ratio de {di_ratio:.2f} viola o limiar legal de 80% (Four-Fifths Rule). "
                        f"O grupo '{unpriv_key}' recebe menos decisões favoráveis que o grupo '{priv_key}'."
                    ),
                    action_item=(
                        f"Atribuir pesos amostrais maiores aos exemplos positivos do grupo '{unpriv_key}' "
                        f"no dataset de treino para equilibrar as frequências marginais."
                    ),
                )
            )
            recommendations.append(
                MitigationRecommendation(
                    stage="inprocessing",
                    title="Restrição de Paridade Demográfica no Treinamento",
                    description="Otimização de hiperparâmetros com penalidade explícita de disparidade estatística.",
                    action_item="Adicionar restrição de Demographic Parity como critério de parada na busca de modelos.",
                )
            )

        if not is_eq_opp_compliant:
            recommendations.append(
                MitigationRecommendation(
                    stage="postprocessing",
                    title="Otimização de Limiares Específicos por Grupo (Equalized Odds Post-Processing)",
                    description=(
                        f"Diferença de Igualdade de Oportunidade de {eq_opp_diff:.2f} excede {equal_opportunity_threshold:.2f}. "
                        f"As taxas de acerto positivo (TPR) diferem significativamente entre os grupos."
                    ),
                    action_item=(
                        f"Ajustar limiar de decisão de probabilidade separadamente para '{unpriv_key}' e '{priv_key}' "
                        f"para equalizar as taxas de verdadeiros positivos e falsos positivos."
                    ),
                )
            )
    else:
        recommendations.append(
            MitigationRecommendation(
                stage="monitoring",
                title="Monitoramento Contínuo de Viés em Produção",
                description="O modelo atende a todos os critérios regulatórios de justiça algorítmica avaliados.",
                action_item="Manter auditoria periódica em lotes de inferência produtivos para prevenir deriva ética.",
            )
        )

    return FairnessAuditReport(
        protected_attribute=attribute_name,
        total_samples=len(yt),
        group_metrics={k: asdict(v) for k, v in group_metrics_map.items()},
        fairness_metrics=fairness_metrics,
        status=overall_status,
        recommendations=recommendations,
    )
