"""Estado compartilhado pelas fatias de ``MLService`` (issue #9)."""

from __future__ import annotations

import re
from typing import Any

from ..config import Settings
from ..experiment_tracking import ExperimentTracker
from ..observability import get_logger
from ..registry import ModelRegistry
from ..serving.onnx_backend import OnnxPredictor

logger = get_logger()

MONITORING_WINDOW = 500
MIN_DRIFT_SAMPLES = 30
IRIS_DATASET = "iris"
COMPARISON_RUN_ID = re.compile(r"\d{8}T\d{6}-[0-9a-f]{8}")


class ServiceBase:
    """Atributos que ``MLService.__init__`` inicializa e todas as fatias usam."""

    settings: Settings
    registry: ModelRegistry
    tracker: ExperimentTracker
    _artifacts: dict[str, tuple[tuple[str, int], dict[str, Any]]]
    _onnx_sessions: dict[str, tuple[tuple[str | None, int], OnnxPredictor]]
