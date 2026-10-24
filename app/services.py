"""Fachada do serviço de ML.

``MLService`` é composto por fatias por domínio em ``app.ml_service`` (issue #9);
este módulo mantém os nomes públicos de antes por compatibilidade.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .config import Settings, get_settings
from .experiment_tracking import ExperimentTracker
from .ml import load_artifact, predict_artifact, train_model
from .ml_service._base import COMPARISON_RUN_ID, IRIS_DATASET, MIN_DRIFT_SAMPLES, MONITORING_WINDOW, logger
from .ml_service.canary import CanaryMixin
from .ml_service.comparison import ComparisonMixin
from .ml_service.data import DataMixin
from .ml_service.datasets import DatasetsMixin
from .ml_service.inference import InferenceMixin
from .ml_service.monitoring import MonitoringMixin
from .ml_service.registry import RegistryMixin
from .ml_service.training import TrainingMixin
from .platform.cache import prediction_cache
from .registry import ModelRegistry
from .serving.onnx_backend import OnnxPredictor

__all__ = [
    "COMPARISON_RUN_ID",
    "IRIS_DATASET",
    "MIN_DRIFT_SAMPLES",
    "MONITORING_WINDOW",
    "MLService",
    "get_service",
    "load_artifact",
    "logger",
    "predict_artifact",
    "prediction_cache",
    "train_model",
]


class MLService(
    RegistryMixin,
    DatasetsMixin,
    ComparisonMixin,
    CanaryMixin,
    TrainingMixin,
    InferenceMixin,
    DataMixin,
    MonitoringMixin,
):
    """Serviço de ML: registry, datasets, comparação, canário, treino, inferência e monitoramento."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.registry = ModelRegistry(settings.model_dir, settings.artifact_signing_key)
        self.tracker = ExperimentTracker(settings.mlflow_tracking_uri)
        # ponytail: cache sem limite, uma entrada por arquivo de artefato já servido.
        self._artifacts: dict[str, tuple[tuple[str, int], dict[str, Any]]] = {}
        self._onnx_sessions: dict[str, tuple[tuple[str | None, int], OnnxPredictor]] = {}


@lru_cache
def get_service() -> MLService:
    """Instância única do serviço, compartilhada por todos os routers e pela CLI."""
    return MLService(get_settings())
