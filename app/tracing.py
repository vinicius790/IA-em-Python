"""Tracing distribuído com OpenTelemetry, opcional (extra `otel`).

Sem o extra instalado tudo aqui vira no-op: a API sobe igual e nenhum import do
OpenTelemetry acontece fora deste módulo. Com o extra:

* `setup_tracing(app, engine)` instrumenta FastAPI e SQLAlchemy (eventos do engine);
* `traced(nome)` abre um span em volta de treino, predição, drift e promoção;
* `set_span_attributes(...)` anota o span corrente (ex.: `model_version`, `dataset_sha`);
* `current_trace_id()` liga o trace aos logs estruturados e aos eventos de auditoria.

O exportador OTLP só é ligado quando `OTEL_EXPORTER_OTLP_ENDPOINT` (ou
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) está definida; por padrão nada sai do processo.
"""

from __future__ import annotations

import functools
import importlib.util
import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

TRACER_NAME = "iris-ai"
OTLP_ENV_VARS = ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT")


def otel_available() -> bool:
    """True quando a API do OpenTelemetry está instalada."""
    return importlib.util.find_spec("opentelemetry") is not None


def otlp_endpoint() -> str | None:
    """Endpoint OTLP configurado por ambiente; `None` mantém a exportação desligada."""
    for name in OTLP_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _trace_api() -> Any:
    if not otel_available():
        return None
    from opentelemetry import trace  # pylint: disable=import-outside-toplevel,import-error

    return trace


def current_trace_id() -> str | None:
    """trace_id (hex, 32 caracteres) do span corrente, ou `None` sem trace ativo."""
    trace = _trace_api()
    if trace is None:
        return None
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")


def set_span_attributes(**attributes: Any) -> None:
    """Anota o span corrente; valores `None` são ignorados e os demais viram str/int/float/bool."""
    trace = _trace_api()
    if trace is None:
        return
    span = trace.get_current_span()
    if not span.is_recording():
        return
    for key, value in attributes.items():
        if value is None:
            continue
        if not isinstance(value, str | bool | int | float):
            value = str(value)
        span.set_attribute(f"iris.{key}", value)


def traced(name: str) -> Callable[[F], F]:
    """Decorador: executa a função dentro de um span `name` (no-op sem o extra)."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = _trace_api()
            if trace is None:
                return func(*args, **kwargs)
            with trace.get_tracer(TRACER_NAME).start_as_current_span(name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def add_trace_id(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Processor do structlog: inclui `trace_id` quando há um trace ativo."""
    trace_id = current_trace_id()
    if trace_id is not None:
        event_dict.setdefault("trace_id", trace_id)
    return event_dict


def setup_tracing(app: Any, engine: Any, *, service_name: str = TRACER_NAME) -> bool:
    """Instrumenta FastAPI e SQLAlchemy quando o extra `otel` está instalado.

    Um `TracerProvider` com exportador OTLP só é registrado quando há endpoint configurado
    e nenhum provider foi definido antes (por testes ou pelo `opentelemetry-instrument`).
    Retorna True quando a instrumentação foi aplicada.
    """
    # pylint: disable=import-outside-toplevel,import-error
    if not otel_available():
        return False
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        return False
    if otlp_endpoint() is not None:
        _install_otlp_provider(service_name)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,health")
    instrument_engine(engine)
    return True


def instrument_engine(engine: Any) -> None:
    """Um span `db.<VERBO>` por statement SQL, via eventos do SQLAlchemy.

    Usa os eventos públicos do engine em vez de `opentelemetry-instrumentation-sqlalchemy`,
    que ainda não aceita SQLAlchemy 2.1. Idempotente por engine.
    """
    # pylint: disable=import-outside-toplevel,import-error
    from opentelemetry import trace
    from sqlalchemy import event

    if getattr(engine, "_iris_traced", False):
        return
    tracer = trace.get_tracer(TRACER_NAME)
    system = engine.dialect.name

    def before(_conn: Any, _cursor: Any, statement: str, _params: Any, context: Any, _many: bool) -> None:
        verb = statement.lstrip().split(" ", 1)[0].upper() or "SQL"
        span = tracer.start_span(
            f"db.{verb}",
            kind=trace.SpanKind.CLIENT,
            attributes={"db.system": system, "db.statement": statement[:1000]},
        )
        context._iris_span = span  # pylint: disable=protected-access

    def after(_conn: Any, _cursor: Any, _statement: str, _params: Any, context: Any, _many: bool) -> None:
        span = getattr(context, "_iris_span", None)
        if span is not None:
            span.end()

    def on_error(exception_context: Any) -> None:
        span = getattr(exception_context.execution_context, "_iris_span", None)
        if span is not None:
            span.record_exception(exception_context.original_exception)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            span.end()

    event.listen(engine, "before_cursor_execute", before)
    event.listen(engine, "after_cursor_execute", after)
    event.listen(engine, "handle_error", on_error)
    engine._iris_traced = True  # pylint: disable=protected-access


def _install_otlp_provider(service_name: str) -> None:
    # pylint: disable=import-outside-toplevel,import-error
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
