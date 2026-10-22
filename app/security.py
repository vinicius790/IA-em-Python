from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from time import monotonic
from uuid import uuid4

from fastapi import Header, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import get_settings
from .database import SessionLocal
from .db_models import ApiUser
from .governance.rbac import (
    ANONYMOUS,
    MASTER,
    Principal,
    can,
    current_principal,
    hash_key,
    required_role,
    same_key,
)

_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def _user_for_key(key: str) -> Principal | None:
    # ponytail: uma consulta indexada por requisição autenticada com chave de usuário;
    # se isso pesar, coloque um cache com expiração curta na frente.
    with SessionLocal() as db:
        user = db.scalar(select(ApiUser).where(ApiUser.key_hash == hash_key(key), ApiUser.active.is_(True)))
    return Principal(user.name, user.role) if user else None


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> Principal:
    """Única dependência de autenticação e autorização da API.

    Com `API_KEY` vazia (permitido só em `ENVIRONMENT=development`) a API fica aberta.
    A chave mestra `API_KEY` (comparada em tempo constante) é `admin`; as demais chaves
    são as dos usuários de `api_users`, cada uma com o seu papel. Leitura exige `reader`,
    o resto exige `operator`; rotas de administração pedem `admin` com `require_role`.

    É `async` de propósito: assim o `current_principal` vale para o resto da requisição
    (uma dependência síncrona rodaria em outra thread, com uma cópia do contexto).
    """
    expected = get_settings().api_key
    if not expected:
        principal = ANONYMOUS
    elif x_api_key and same_key(x_api_key, expected):
        principal = MASTER
    else:
        principal = await run_in_threadpool(_user_for_key, x_api_key) if x_api_key else None
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
    needed = required_role(request.method, request.url.path)
    if not can(principal.role, needed):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, f"Role '{principal.role}' cannot do this; requires '{needed}'"
        )
    request.state.principal = principal
    current_principal.set(principal)
    return principal


def require_role(role: str) -> Callable[[Request], Principal]:
    """Dependência de rota que exige um papel mínimo (use depois de `require_api_key`)."""

    def dependency(request: Request) -> Principal:
        principal: Principal = getattr(request.state, "principal", ANONYMOUS)
        if not can(principal.role, role):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, f"Role '{principal.role}' cannot do this; requires '{role}'"
            )
        return principal

    return dependency


def sanitize_request_id(value: str | None) -> str:
    """Aceita o X-Request-ID do cliente só se for curto e sem caracteres de controle."""
    if value and _REQUEST_ID.fullmatch(value):
        return value
    return uuid4().hex


class RateLimiter:
    """Janela fixa por chave, em memória.

    ponytail: estado local ao processo; com mais de uma réplica, troque por um
    limitador compartilhado (Redis) ou delegue ao gateway.
    """

    def __init__(self, limit: int = 120, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._buckets: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()
        self._next_purge = monotonic() + window_seconds

    def allow(self, key: str) -> bool:
        now = monotonic()
        with self._lock:
            if now >= self._next_purge:
                self._purge(now)
            started_at, count = self._buckets.get(key, (now, 0))
            if now - started_at >= self.window_seconds:
                started_at, count = now, 0
            self._buckets[key] = (started_at, count + 1)
            return count + 1 <= self.limit

    def _purge(self, now: float) -> None:
        self._buckets = {
            key: bucket for key, bucket in self._buckets.items() if now - bucket[0] < self.window_seconds
        }
        self._next_purge = now + self.window_seconds

    def __len__(self) -> int:
        return len(self._buckets)


class BodySizeLimitMiddleware:
    """Recusa corpos maiores que `max_bytes`, contando os bytes realmente recebidos.

    O header Content-Length serve só de atalho: corpos em `chunked` (sem o header)
    também são medidos.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            await _send_too_large(send)
            return
        received = 0
        rejected = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    rejected = True
                    raise _BodyTooLargeError
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if rejected:
                return  # descarta a reação da aplicação à leitura interrompida (ex.: 400 do FastAPI)
            response_started = response_started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not rejected:
                raise
        if rejected and not response_started:
            await _send_too_large(send)


class _BodyTooLargeError(Exception):
    """Interrompe a leitura do corpo quando o limite é ultrapassado."""


async def _send_too_large(send: Send) -> None:
    body = json.dumps({"detail": "Request body too large"}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status.HTTP_413_CONTENT_TOO_LARGE,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
