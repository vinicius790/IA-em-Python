from __future__ import annotations

from typing import Any


class ExperimentTracker:
    """Envia a execução ao MLflow quando ele está instalado e `MLFLOW_TRACKING_URI` definido.

    Sem MLflow o backend é `database`: quem chama grava a execução na tabela
    `experiment_runs`, que é o registro local de experimentos.
    """

    def __init__(self, tracking_uri: str = "") -> None:
        self.mlflow = None
        if tracking_uri:
            try:
                import mlflow
            except ImportError:
                mlflow = None
            if mlflow is not None:
                mlflow.set_tracking_uri(tracking_uri)
                self.mlflow = mlflow

    @property
    def backend(self) -> str:
        return "mlflow" if self.mlflow else "database"

    def log_run(
        self, name: str, params: dict[str, Any], metrics: dict[str, float], tags: dict[str, str] | None = None
    ) -> dict[str, Any]:
        record: dict[str, Any] = {"name": name, "params": params, "metrics": metrics, "tags": tags or {}}
        if self.mlflow:
            with self.mlflow.start_run(run_name=name) as run:
                self.mlflow.log_params({key: str(value) for key, value in params.items()})
                self.mlflow.log_metrics(metrics)
                self.mlflow.set_tags(tags or {})
                record["run_id"] = run.info.run_id
        return record
