"""Motor de Validação Cruzada K-Fold e Estratificada com Métricas Avançadas e Intervalos de Confiança."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from .intervals import confidence_interval


@dataclass
class KFoldValidationResult:
    """Resultado consolidado da validação cruzada K-Fold."""

    model_name: str
    n_splits: int
    stratified: bool
    sample_count: int
    classes: List[Any]
    feature_names: List[str]
    fold_metrics: List[Dict[str, Any]]
    aggregated_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_kfold_cross_validation(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[Sequence[Any], np.ndarray, pd.Series],
    model_name: str = "model",
    n_splits: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    random_state: int = 42,
    feature_names: Optional[List[str]] = None,
) -> KFoldValidationResult:
    """Executa validação cruzada K-Fold (ou Estratificada) com rastreamento completo de métricas.

    Gera métricas por fold, média, desvio padrão, extremos e intervalos de confiança
    bilaterais a 95%, além de matriz de confusão out-of-fold.
    """
    if isinstance(X, pd.DataFrame):
        feat_names = list(X.columns)
        X_mat = X.to_numpy()
    else:
        X_mat = np.asarray(X)
        feat_names = feature_names or [f"feature_{i}" for i in range(X_mat.shape[1])]

    y_arr = np.asarray(y)
    classes = list(np.unique(y_arr))
    sample_count = len(y_arr)

    if n_splits < 2:
        raise ValueError("n_splits deve ser pelo menos 2.")
    if n_splits > sample_count:
        raise ValueError(f"n_splits ({n_splits}) não pode ser maior que o número de amostras ({sample_count}).")

    if stratified and len(classes) > 1:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    fold_metrics: List[Dict[str, Any]] = []
    oof_preds = np.empty_like(y_arr)
    metric_accumulators: Dict[str, List[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "precision_macro": [],
        "recall_macro": [],
        "f1_macro": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_mat, y_arr)):
        X_train, X_val = X_mat[train_idx], X_mat[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        # Instanciar fold limpo do estimador
        fold_model = clone(model)
        if isinstance(X, pd.DataFrame):
            X_train_df = pd.DataFrame(X_train, columns=feat_names)
            X_val_df = pd.DataFrame(X_val, columns=feat_names)
            fold_model.fit(X_train_df, y_train)
            preds = fold_model.predict(X_val_df)
        else:
            fold_model.fit(X_train, y_train)
            preds = fold_model.predict(X_val)

        oof_preds[val_idx] = preds

        acc = float(accuracy_score(y_val, preds))
        bal_acc = float(balanced_accuracy_score(y_val, preds))
        prec = float(precision_score(y_val, preds, average="macro", zero_division=0))
        rec = float(recall_score(y_val, preds, average="macro", zero_division=0))
        f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

        metric_accumulators["accuracy"].append(acc)
        metric_accumulators["balanced_accuracy"].append(bal_acc)
        metric_accumulators["precision_macro"].append(prec)
        metric_accumulators["recall_macro"].append(rec)
        metric_accumulators["f1_macro"].append(f1)

        fold_metrics.append(
            {
                "fold": fold_idx + 1,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "precision_macro": round(prec, 4),
                "recall_macro": round(rec, 4),
                "f1_macro": round(f1, 4),
            }
        )

    # Cálculo dos agregados e intervalos de confiança a 95%
    aggregated_metrics: Dict[str, Dict[str, float]] = {}
    for metric_name, values in metric_accumulators.items():
        ci = confidence_interval(values, confidence=0.95)
        aggregated_metrics[metric_name] = {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values, ddof=1) if len(values) > 1 else 0.0), 4),
            "median": round(float(np.median(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
            "ci_lower": round(ci["lower"], 4),
            "ci_upper": round(ci["upper"], 4),
        }

    cm = confusion_matrix(y_arr, oof_preds, labels=classes).tolist()

    return KFoldValidationResult(
        model_name=model_name,
        n_splits=n_splits,
        stratified=stratified,
        sample_count=sample_count,
        classes=classes,
        feature_names=feat_names,
        fold_metrics=fold_metrics,
        aggregated_metrics=aggregated_metrics,
        confusion_matrix=cm,
    )
