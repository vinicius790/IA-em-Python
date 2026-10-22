"""Autenticação, middleware e limites: regressões das falhas de segurança da 7.0."""

import asyncio
import re

import pytest
from prometheus_client import generate_latest
from pydantic import ValidationError

from app import main
from app.config import Settings
from app.main import app
from app.security import BodySizeLimitMiddleware, RateLimiter, sanitize_request_id
from tests.conftest import SETOSA

PUBLIC_PATHS = {"/healthz", "/readyz", "/api/v1/health", "/api/v1/ready"}


def _api_operations() -> list[tuple[str, str]]:
    return [
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if path not in PUBLIC_PATHS
    ]


def test_every_api_route_requires_the_key(client, api_key):
    operations = _api_operations()
    assert len(operations) > 40
    for method, path in operations:
        url = re.sub(r"\{(iris|prediction|canary|user)_id\}", "1", path).replace("{", "").replace("}", "")
        response = client.request(method, url)
        assert response.status_code == 401, f"{method} {path} respondeu {response.status_code} sem API key"


def test_wrong_key_is_rejected_and_right_key_is_accepted(client, api_key):
    assert client.get("/api/v1/models", headers={"X-API-Key": "errada"}).status_code == 401
    assert client.get("/api/v1/models", headers={"X-API-Key": api_key}).status_code == 200


def test_public_probes_stay_open_with_auth_enabled(client, api_key):
    for path in [*PUBLIC_PATHS, "/api/v1/metrics", "/", "/console/", "/docs"]:
        assert client.get(path).status_code == 200, path


def test_api_is_open_only_without_configured_key(client):
    assert client.get("/api/v1/models").status_code == 200


def test_settings_refuse_to_start_without_key_outside_development():
    with pytest.raises(ValidationError, match="API_KEY"):
        Settings(_env_file=None, environment="production", api_key="")
    assert Settings(_env_file=None, environment="production", api_key="k").api_key == "k"


def test_arbitrary_sql_endpoint_is_gone(client):
    response = client.post("/api/v2/data/query", json={"path": "x.csv", "sql": "select 1"})
    assert response.status_code == 404
    assert not any("query" in path for path in app.openapi()["paths"])


def test_repository_snapshot_endpoint_is_gone(client):
    assert client.get("/api/v6/foundry/manifest").status_code == 404


@pytest.mark.parametrize("value", ["abc-123_X.y", "a" * 64])
def test_request_id_accepts_safe_values(value):
    assert sanitize_request_id(value) == value


@pytest.mark.parametrize("value", [None, "", "a" * 65, "com espaço", "quebra\nlinha", "<script>"])
def test_request_id_replaces_unsafe_values(value):
    generated = sanitize_request_id(value)
    assert generated != value
    assert len(generated) == 32


def test_request_id_header_round_trip(client):
    assert (
        client.get("/healthz", headers={"X-Request-ID": "pedido-42"}).headers["X-Request-ID"] == "pedido-42"
    )
    echoed = client.get("/healthz", headers={"X-Request-ID": "x" * 500}).headers["X-Request-ID"]
    assert len(echoed) == 32


def test_rate_limiter_blocks_after_limit_and_purges_old_buckets(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("app.security.monotonic", lambda: clock[0])
    limiter = RateLimiter(limit=2, window_seconds=10)
    assert [limiter.allow("a") for _ in range(3)] == [True, True, False]
    for index in range(50):
        limiter.allow(f"cliente-{index}")
    assert len(limiter) == 51
    clock[0] = 11.0
    assert limiter.allow("a") is True
    assert len(limiter) == 1, "baldes vencidos precisam ser descartados"


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(main, "rate_limiter", RateLimiter(limit=1, window_seconds=60))
    assert client.get("/healthz").status_code == 200
    blocked = client.get("/healthz")
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "60"


def test_body_limit_rejects_declared_oversize(client):
    response = client.post(
        "/api/v1/predict", content=b"x" * 2_100_000, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


def test_body_limit_counts_bytes_without_content_length(client):
    def chunks():
        for _ in range(21):
            yield b"x" * 100_000

    response = client.post("/api/v1/predict", content=chunks(), headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_body_limit_middleware_counts_received_bytes():
    async def downstream(scope, receive, send):
        while (await receive()).get("more_body"):
            pass
        await send({"type": "http.response.start", "status": 200, "headers": []})

    def run(chunks: list[bytes]) -> list[dict]:
        messages = [
            {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
            for index, chunk in enumerate(chunks)
        ]
        sent: list[dict] = []

        async def receive():
            return messages.pop(0)

        async def send(message):
            sent.append(message)

        middleware = BodySizeLimitMiddleware(downstream, max_bytes=10)
        asyncio.run(middleware({"type": "http", "headers": []}, receive, send))
        return sent

    assert run([b"x" * 5, b"x" * 5])[0]["status"] == 200
    rejected = run([b"x" * 6, b"x" * 6, b"x" * 6])
    assert [message["type"] for message in rejected] == ["http.response.start", "http.response.body"]
    assert rejected[0]["status"] == 413


def test_metrics_path_label_uses_route_template(client):
    for iris_id in (910001, 910002, 910003):
        client.get(f"/api/v1/iris/{iris_id}")
        client.get(f"/sondagem/{iris_id}")
    exposition = generate_latest().decode()
    assert 'path="/api/v1/iris/{iris_id}"' in exposition
    assert 'path="unmatched"' in exposition
    assert "910001" not in exposition


def test_unhandled_errors_return_500_with_request_id(client, monkeypatch):
    def boom():
        raise RuntimeError("detalhe interno")

    monkeypatch.setattr("app.api_v2.service.quality_report", boom)
    response = client.get("/api/v2/data/quality")
    assert response.status_code == 500
    assert "detalhe interno" not in response.text
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_prediction_payload_is_validated(client):
    assert client.post("/api/v1/predict", json={**SETOSA, "sepal_length": -1}).status_code == 422
    assert client.post("/api/v1/predict", json={**SETOSA, "extra": 1}).status_code == 422
