from __future__ import annotations

from collections.abc import Callable
from typing import Any


def optimize_xgboost(
    factory: Callable[[dict[str, Any]], Any], X, y, *, seed: int = 42, n_trials: int = 12, cv: int = 3
) -> dict[str, Any] | None:
    """Busca TPE com Optuna. Devolve `None` se o Optuna não estiver instalado."""
    try:
        import optuna
    except ImportError:
        return None
    from sklearn.model_selection import cross_val_score

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        }
        return float(cross_val_score(factory(params), X, y, cv=cv, scoring="f1_macro").mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "method": "optuna-tpe",
        "best_params": dict(study.best_params),
        "best_score": round(float(study.best_value), 8),
        "trials": len(study.trials),
    }
