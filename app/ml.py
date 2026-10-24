from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import prod
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import __version__
from .data import FEATURE_COLUMNS, TARGET_COLUMN
from .explainability import permutation_explanation
from .lineage import dataset_fingerprint, lineage_record
from .model_card import create_model_card
from .monitoring.calibration import multiclass_brier
from .optimization import optimize_xgboost
from .quality import validate_training_frame

BACKGROUND_ROWS = 100
LATENCY_SAMPLES = 20


@dataclass(slots=True)
class TrainingResult:
    model_name: str
    model: Any
    metrics: dict[str, Any]
    artifact_path: str
    duration_seconds: float
    dataset_hash: str
    lineage_hash: str
    model_card_path: str


def _pipeline(estimator: Any, *, scale: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


class _CatBoostAdapter(ClassifierMixin, BaseEstimator):
    """Adaptador scikit-learn para o CatBoost (extra opcional `boosters`)."""

    def __init__(
        self, iterations: int = 300, depth: int = 6, learning_rate: float = 0.05, random_seed: int = 42
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed

    def fit(self, X, y):
        from catboost import CatBoostClassifier

        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function="MultiClass",
            verbose=False,
            random_seed=self.random_seed,
            allow_writing_files=False,
        )
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).reshape(-1).astype(int)

    def predict_proba(self, X):
        return np.asarray(self.model_.predict_proba(X))


def _optional_boosters(random_state: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        from lightgbm import LGBMClassifier

        result["lightgbm"] = _pipeline(
            LGBMClassifier(n_estimators=300, learning_rate=0.04, verbosity=-1, random_state=random_state),
            scale=False,
        )
    except ImportError:
        pass
    try:
        import catboost  # noqa: F401  # pylint: disable=unused-import

        result["catboost"] = _pipeline(_CatBoostAdapter(random_seed=random_state), scale=False)
    except ImportError:
        pass
    return result


def _xgboost(random_state: int, **params: Any) -> Pipeline:
    defaults = {
        "n_estimators": 350,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }
    return _pipeline(
        XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            **(defaults | params),
        ),
        scale=False,
    )


def build_models(random_state: int = 42) -> dict[str, Any]:
    """Catálogo de modelos: nome -> estimador scikit-learn ainda não treinado."""
    rs = random_state
    models: dict[str, Any] = {
        "logistic_regression": _pipeline(LogisticRegression(max_iter=2500, random_state=rs)),
        "knn": _pipeline(KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "naive_bayes": _pipeline(GaussianNB()),
        "svm": _pipeline(CalibratedClassifierCV(SVC(random_state=rs), ensemble=False)),
        "decision_tree": _pipeline(DecisionTreeClassifier(max_depth=6, random_state=rs), scale=False),
        "random_forest": _pipeline(
            RandomForestClassifier(n_estimators=400, random_state=rs, n_jobs=-1), scale=False
        ),
        "extra_trees": _pipeline(
            ExtraTreesClassifier(n_estimators=400, random_state=rs, n_jobs=-1), scale=False
        ),
        "gradient_boosting": _pipeline(
            GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, random_state=rs), scale=False
        ),
        "hist_gradient_boosting": _pipeline(
            HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, random_state=rs), scale=False
        ),
        "adaboost": _pipeline(
            AdaBoostClassifier(n_estimators=250, learning_rate=0.05, random_state=rs), scale=False
        ),
        "linear_sgd": _pipeline(
            CalibratedClassifierCV(
                SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, random_state=rs), cv=3
            )
        ),
        "linear_discriminant": _pipeline(LinearDiscriminantAnalysis()),
        "quadratic_discriminant": _pipeline(QuadraticDiscriminantAnalysis()),
        "xgboost": _xgboost(rs),
        "mlp": _pipeline(
            MLPClassifier(
                hidden_layer_sizes=(128, 64, 32, 16),
                alpha=1e-4,
                max_iter=1600,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,
                random_state=rs,
            )
        ),
    }
    base_estimators = [("rf", models["random_forest"]), ("svm", models["svm"]), ("xgb", models["xgboost"])]
    models["voting_ensemble"] = VotingClassifier(estimators=base_estimators, voting="soft")
    models["stacking_ensemble"] = StackingClassifier(
        estimators=base_estimators[:2],
        final_estimator=LogisticRegression(max_iter=2500, random_state=rs),
        passthrough=True,
    )
    models.update(_optional_boosters(rs))
    return models


def available_models() -> list[str]:
    return sorted(build_models())


def get_model(name: str, random_state: int = 42) -> Any:
    """Retorna uma instância limpa de estimador do catálogo."""
    models = build_models(random_state=random_state)
    if name not in models:
        raise ValueError(f"Modelo desconhecido: {name}. Opções: {list(models.keys())}")
    return models[name]


def _score_dict(y_true, y_pred) -> dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
    }


TUNING_SPACES: dict[str, dict[str, list[Any]]] = {
    "xgboost": {
        "model__max_depth": [2, 3, 4, 5, 6, 8],
        "model__learning_rate": [0.02, 0.03, 0.05, 0.1, 0.2],
        "model__n_estimators": [150, 250, 350, 500],
        "model__subsample": [0.7, 0.85, 1.0],
    },
    "random_forest": {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [None, 4, 6, 8, 12],
        "model__min_samples_split": [2, 3, 4, 6],
        "model__max_features": ["sqrt", "log2", None],
    },
    "svm": {"model__estimator__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
    "logistic_regression": {"model__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
    "mlp": {
        "model__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
        "model__alpha": [1e-5, 1e-4, 1e-3],
        "model__learning_rate_init": [3e-4, 1e-3, 3e-3],
    },
}


def tune_model(
    model_name: str, model: Any, X, y, *, random_state: int, cv_folds: int, trials: int
) -> tuple[Any, dict[str, Any]]:
    """Devolve o modelo (não treinado) com os melhores hiperparâmetros e o resumo da busca.

    XGBoost usa Optuna quando instalado; sem Optuna, e para os demais modelos com
    espaço de busca, usa RandomizedSearchCV. Modelo sem espaço de busca é erro
    explícito: pedir tuning nunca passa em silêncio.
    """
    trials = max(1, trials)
    if model_name == "xgboost":
        result = optimize_xgboost(
            lambda params: _xgboost(random_state, **params),
            X,
            y,
            seed=random_state,
            n_trials=trials,
            cv=cv_folds,
        )
        if result is not None:
            return _xgboost(random_state, **result["best_params"]), result
    space = TUNING_SPACES.get(model_name)
    if space is None:
        raise ValueError(
            f"Model '{model_name}' has no hyperparameter search space. "
            f"Tunable models: {', '.join(sorted(TUNING_SPACES))}"
        )
    n_iter = min(trials, prod(len(values) for values in space.values()))
    search = RandomizedSearchCV(
        model,
        space,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring="f1_macro",
        random_state=random_state,
        refit=False,
    ).fit(X, y)
    summary = {
        "method": "randomized-search",
        "best_params": search.best_params_,
        "best_score": round(float(search.best_score_), 8),
        "trials": n_iter,
    }
    return clone(model).set_params(**search.best_params_), summary


def _single_row_latency_ms(model: Any, X: pd.DataFrame) -> float:
    """p95 do tempo de `predict_proba` para uma linha, medido no modelo treinado."""
    timings = []
    for position in range(min(LATENCY_SAMPLES, len(X))):
        row = X.iloc[[position]]
        started = perf_counter()
        model.predict_proba(row)
        timings.append((perf_counter() - started) * 1000)
    return round(float(np.percentile(timings, 95)), 4)


def train_model(
    frame: pd.DataFrame,
    model_name: str,
    artifact_dir: str | Path,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    cv_folds: int = 5,
    tune_hyperparameters: bool = False,
    optimization_trials: int = 12,
    features: list[str] | None = None,
    target: str | None = None,
    dataset: str | None = None,
) -> TrainingResult:
    """Treina `model_name` em `frame`.

    Sem `features`/`target` o treino é o do Iris. Com um dataset do catálogo, o nome
    registrado passa a ser `<dataset>.<modelo>`, para não misturar versões de dados diferentes.
    """
    start = perf_counter()
    quality = validate_training_frame(frame, features, target)
    features = list(features or FEATURE_COLUMNS)
    target = target or TARGET_COLUMN
    if not quality.passed:
        failed = [item["name"] for item in quality.checks if not item["passed"]]
        raise ValueError(f"Dataset quality gates failed: {', '.join(failed)}")

    models = build_models(random_state)
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(sorted(models))}")

    X = frame[features]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(frame[target])
    classes = [str(value) for value in label_encoder.classes_]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model: Any = models[model_name]
    algorithm = model_name
    optimization: dict[str, Any] | None = None
    if tune_hyperparameters:
        model, optimization = tune_model(
            algorithm,
            model,
            X_train,
            y_train,
            random_state=random_state,
            cv_folds=cv_folds,
            trials=optimization_trials,
        )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
        return_train_score=True,
    )
    model.fit(X_train, y_train)
    probabilities = np.asarray(model.predict_proba(X_test))
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred = label_encoder.inverse_transform(np.asarray(model.predict(X_test)).astype(int))
    metrics = _score_dict(y_test_labels, y_pred)
    metrics.update(
        {
            "cv_accuracy_mean": round(float(cv_results["test_accuracy"].mean()), 6),
            "cv_accuracy_std": round(float(cv_results["test_accuracy"].std()), 6),
            "cv_f1_macro_mean": round(float(cv_results["test_f1_macro"].mean()), 6),
            "cv_balanced_accuracy_mean": round(float(cv_results["test_balanced_accuracy"].mean()), 6),
            "train_accuracy_mean": round(float(cv_results["train_accuracy"].mean()), 6),
            "generalization_gap": round(
                float(cv_results["train_accuracy"].mean() - cv_results["test_accuracy"].mean()), 6
            ),
            "brier_score": round(multiclass_brier(y_test_labels, probabilities, classes), 6),
            "latency_ms": _single_row_latency_ms(model, X_test),
            "feature_importance": permutation_explanation(model, X_test, y_test, random_state),
            "classification_report": classification_report(
                y_test_labels, y_pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test_labels, y_pred, labels=label_encoder.classes_
            ).tolist(),
            "classes": classes,
            "dataset_quality": quality.score,
        }
    )
    if optimization:
        metrics["optimization"] = optimization

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if dataset:
        model_name = f"{dataset}.{algorithm}"
    dataset_hash = dataset_fingerprint(frame)
    parameters = {
        "random_state": random_state,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "tune_hyperparameters": tune_hyperparameters,
        "optimization_trials": optimization_trials,
        "dataset": dataset or "iris",
        "algorithm": algorithm,
    }
    lineage = lineage_record(
        model_name=model_name, dataset_hash=dataset_hash, parameters=parameters, metrics=metrics
    )
    model_card = create_model_card(model_name, metrics, parameters, classes, dataset or "iris")

    fingerprint = hashlib.sha256(
        json.dumps([model_name, lineage["lineage_hash"]], sort_keys=True).encode()
    ).hexdigest()[:16]
    artifact_path = artifact_dir / f"{model_name}_{fingerprint}.joblib"
    card_path = artifact_dir / f"{model_name}_{fingerprint}.model-card.json"
    lineage_path = artifact_dir / f"{model_name}_{fingerprint}.lineage.json"
    background = X_train.sample(min(BACKGROUND_ROWS, len(X_train)), random_state=random_state)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "target": target,
            "label_encoder": label_encoder,
            "background": background.reset_index(drop=True),
            "metadata": {
                "artifact_version": __version__,
                "dataset": dataset or "iris",
                "dataset_hash": dataset_hash,
                "lineage_hash": lineage["lineage_hash"],
            },
        },
        artifact_path,
    )
    card_path.write_text(json.dumps(model_card, indent=2, default=str), encoding="utf-8")
    lineage_path.write_text(json.dumps(lineage, indent=2, default=str), encoding="utf-8")

    return TrainingResult(
        model_name,
        model,
        metrics,
        str(artifact_path),
        round(perf_counter() - start, 4),
        dataset_hash,
        str(lineage["lineage_hash"]),
        str(card_path),
    )


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Desserializa um artefato. Use só depois de `ModelRegistry.verify` (joblib usa pickle)."""
    return joblib.load(path)


def predict_artifact(artifact: dict[str, Any], features: list[list[float]]) -> tuple[list[str], np.ndarray]:
    model = artifact["model"]
    feature_frame = pd.DataFrame(features, columns=artifact["features"])
    probabilities = np.asarray(model.predict_proba(feature_frame))
    encoder: LabelEncoder = artifact["label_encoder"]
    predictions = encoder.inverse_transform(probabilities.argmax(axis=1))
    return [str(value) for value in predictions], probabilities
