"""Tracing OpenTelemetry (extra `otel`): spans, atributos, trace_id em logs e auditoria."""

from __future__ import annotations

import pytest

from app import tracing

from .conftest import SETOSA

pytest.importorskip("opentelemetry.sdk", reason="extra `otel` não instalado")
pytest.importorskip("opentelemetry.instrumentation.fastapi", reason="extra `otel` não instalado")

# pylint: disable=wrong-import-position,wrong-import-order,import-error
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

EXPORTER = InMemorySpanExporter()


def _provider() -> TracerProvider:
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    return provider


PROVIDER = _provider()


@pytest.fixture
def spans():
    EXPORTER.clear()
    yield EXPORTER
    EXPORTER.clear()


def _named(exporter: InMemorySpanExporter, name: str) -> list:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_train_span_has_model_version_and_dataset_sha(client, spans):
    body = client.post("/api/v1/training/train", json={"model_name": "logistic_regression", "cv_folds": 3})
    assert body.status_code == 200, body.text
    (span,) = _named(spans, "iris.train")
    attrs = span.attributes
    assert attrs["iris.model_name"] == "logistic_regression"
    assert attrs["iris.status"] == "completed"
    assert isinstance(attrs["iris.model_version"], int)
    assert attrs["iris.dataset_sha"]
    # o span de domínio é filho do span HTTP do FastAPI e o SQLAlchemy também gera spans
    http = [s for s in spans.get_finished_spans() if s.parent is None]
    assert http and span.context.trace_id == http[0].context.trace_id
    assert any(s.attributes.get("db.system") for s in spans.get_finished_spans())
    events = client.get("/api/v4/audit").json()
    training = next(e for e in events if e["action"] == "training")
    assert training["payload"]["trace_id"] == format(span.context.trace_id, "032x")


def test_predict_span_has_model_version(client, trained, spans):
    response = client.post("/api/v1/predict", json={**SETOSA, "model_name": "logistic_regression"})
    assert response.status_code == 200, response.text
    (span,) = _named(spans, "iris.predict")
    assert span.attributes["iris.model_version"] == response.json()["model_version"]
    assert span.attributes["iris.model_name"] == "logistic_regression"
    assert trained  # garante um modelo servido


def test_trace_id_helpers_inside_and_outside_span():
    assert tracing.current_trace_id() is None
    with trace.get_tracer("t").start_as_current_span("x") as span:
        expected = format(span.get_span_context().trace_id, "032x")
        assert tracing.current_trace_id() == expected
        assert tracing.add_trace_id(None, "info", {})["trace_id"] == expected
    assert "trace_id" not in tracing.add_trace_id(None, "info", {})


def test_otlp_disabled_by_default(monkeypatch):
    for name in tracing.OTLP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert tracing.otlp_endpoint() is None
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    assert tracing.otlp_endpoint() == "http://collector:4318"
