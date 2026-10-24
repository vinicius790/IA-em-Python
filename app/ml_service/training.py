"""Treino de modelos (fatia de ``MLService``, issue #9)."""

from __future__ import annotations

from time import perf_counter

from sqlalchemy.orm import Session

from ..db_models import ExperimentRun, TrainingRun
from ..governance.audit import record_audit
from ..ml import train_model
from ..observability import TRAINING_COUNT
from ..platform.cache import prediction_cache
from ..tracing import set_span_attributes, traced
from ._base import IRIS_DATASET, ServiceBase, logger


class TrainingMixin(ServiceBase):
    @traced("iris.train")
    def train(
        self,
        db: Session,
        model_name: str,
        test_size: float = 0.2,
        cv_folds: int = 5,
        tune_hyperparameters: bool = False,
        optimization_trials: int = 12,
        dataset: str | None = None,
    ) -> TrainingRun:
        """Treina e registra `model_name`. Com `dataset` (nome no catálogo), o modelo é
        registrado como `<dataset>.<modelo>`; sem ele, o treino é o do Iris."""
        dataset = None if dataset in (None, IRIS_DATASET) else dataset
        algorithm = model_name
        model_name = f"{dataset}.{algorithm}" if dataset else algorithm
        parameters = {
            "test_size": test_size,
            "cv_folds": cv_folds,
            "tune_hyperparameters": tune_hyperparameters,
            "optimization_trials": optimization_trials,
            "random_state": self.settings.random_state,
            "dataset": dataset or IRIS_DATASET,
        }
        run = TrainingRun(model_name=model_name, status="running", parameters=parameters)
        db.add(run)
        db.commit()
        db.refresh(run)
        start = perf_counter()
        try:
            # A versão exata do dataset fica no registry: o monitoramento compara contra ela,
            # mesmo depois de um novo envio com o mesmo nome.
            source = self.dataset_record(dataset) if dataset else None
            frame, features, target = self.uploaded_frame(source) if source else self.dataset_frame(None)
            result = train_model(
                frame,
                algorithm,
                self.settings.model_dir,
                random_state=self.settings.random_state,
                test_size=test_size,
                cv_folds=cv_folds,
                tune_hyperparameters=tune_hyperparameters,
                optimization_trials=optimization_trials,
                features=features,
                target=target,
                dataset=dataset,
            )
            registration = self.registry.register(
                model_name,
                result.artifact_path,
                result.metrics,
                metadata={
                    "dataset": dataset or IRIS_DATASET,
                    "dataset_sha256": source["sha256"] if source else None,
                    "dataset_hash": result.dataset_hash,
                    "lineage_hash": result.lineage_hash,
                    "model_card": result.model_card_path,
                    "training_run_id": run.id,
                },
            )
            set_span_attributes(
                model_version=registration["version"],
                dataset=dataset or IRIS_DATASET,
                dataset_sha=source["sha256"] if source else result.dataset_hash,
            )
            run.status = "completed"
            run.metrics = result.metrics
            run.artifact_path = result.artifact_path
            run.duration_seconds = result.duration_seconds
            run.dataset_hash = result.dataset_hash
            run.lineage_hash = result.lineage_hash
            run.model_card_path = result.model_card_path
            experiment = self.tracker.log_run(
                f"{model_name}-{run.id}",
                parameters,
                {k: float(v) for k, v in result.metrics.items() if isinstance(v, int | float)},
                tags={"model": model_name, "version": str(registration["version"])},
            )
            db.add(
                ExperimentRun(
                    name=experiment["name"],
                    backend=self.tracker.backend,
                    params=experiment["params"],
                    metrics=experiment["metrics"],
                    tags=experiment["tags"],
                    external_run_id=experiment.get("run_id"),
                )
            )
            TRAINING_COUNT.labels(model_name, "completed").inc()
            prediction_cache.invalidate(model_name)
        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)
            run.duration_seconds = round(perf_counter() - start, 4)
            TRAINING_COUNT.labels(model_name, "failed").inc()
            logger.exception("training_failed", model=model_name, run_id=run.id)
        db.commit()
        db.refresh(run)
        set_span_attributes(model_name=model_name, run_id=run.id, status=run.status)
        record_audit(db, "training", model=model_name, run_id=run.id, status=run.status)
        return run
