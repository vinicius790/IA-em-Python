from __future__ import annotations

import logging
from time import perf_counter

import structlog
from prometheus_client import Counter, Gauge, Histogram

from .tracing import add_trace_id

REQUEST_COUNT = Counter("iris_api_requests_total", "HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("iris_api_request_duration_seconds", "HTTP request latency", ["method", "path"])
PREDICTION_COUNT = Counter("iris_predictions_total", "Model predictions", ["model", "predicted_class"])
PREDICTION_LATENCY = Histogram("iris_prediction_duration_seconds", "Prediction latency", ["model"])
TRAINING_COUNT = Counter("iris_training_runs_total", "Model training runs", ["model", "status"])
DRIFT_COUNT = Counter("iris_drift_reports_total", "Monitoring snapshots by drift status", ["status"])
ACTIVE_JOBS = Gauge("iris_active_jobs", "Active asynchronous jobs")
HEALTH_SCORE = Gauge("iris_model_health_score", "Health score (0-100) at the last snapshot", ["model"])
DRIFT_PSI = Gauge("iris_model_drift_psi_max", "Largest feature PSI at the last snapshot", ["model"])

UNMATCHED_PATH = "unmatched"


def configure_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(message)s")
    # JSON em ASCII: legível por coletores de log e seguro em consoles que não são UTF-8.
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            add_trace_id,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )


def get_logger():
    return structlog.get_logger("iris-ai")


def record_request(start: float, method: str, route_path: str | None, status: int) -> None:
    """Registra a requisição usando o template da rota, nunca a URL crua.

    URLs cruas dariam cardinalidade ilimitada ao label `path` (um valor por id ou por
    caminho inexistente sondado); o que não casa com rota nenhuma vira `unmatched`.
    """
    path = route_path or UNMATCHED_PATH
    REQUEST_COUNT.labels(method, path, str(status)).inc()
    REQUEST_LATENCY.labels(method, path).observe(round(perf_counter() - start, 4))
