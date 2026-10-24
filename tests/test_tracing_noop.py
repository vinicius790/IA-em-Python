"""Sem o extra `otel` (simulado), o tracing é no-op e não quebra nada."""

from __future__ import annotations

from app import tracing


def test_noop_without_otel(monkeypatch):
    monkeypatch.setattr(tracing, "otel_available", lambda: False)
    assert tracing.current_trace_id() is None
    tracing.set_span_attributes(model_version=1)
    assert tracing.traced("x")(lambda a: a + 1)(1) == 2
    assert tracing.add_trace_id(None, "info", {"event": "e"}) == {"event": "e"}
    assert tracing.setup_tracing(object(), object()) is False
