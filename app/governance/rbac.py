"""Papéis e permissões (só biblioteca padrão).

Três papéis, em ordem crescente: `reader` (só leitura), `operator` (leitura e
operações: treino, predição, upload, canário, promoção) e `admin` (tudo, inclusive
gerenciar usuários). A chave mestra `API_KEY` é sempre `admin`.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from contextvars import ContextVar
from dataclasses import dataclass

ROLES = {"reader": 1, "operator": 2, "admin": 3}
READ_METHODS = {"GET", "HEAD", "OPTIONS"}
# Rotas POST que não alteram estado: um leitor pode predizer, explicar e consultar.
READER_POST_SUFFIXES = (
    "/predict",
    "/predict/batch",
    "/explain",
    "/studio/search",
    "/studio/evaluate-predictions",
    "/security/pii-scan",
    "/features/online-lookup",
    "/governance/fairness-audit",
    "/model/explain",
)
KEY_PREFIX = "iris_"


@dataclass(frozen=True, slots=True)
class Principal:
    name: str
    role: str


ANONYMOUS = Principal("anonymous", "admin")  # API aberta (só em development, sem API_KEY)
MASTER = Principal("api-key", "admin")

# Quem está fazendo a requisição atual; `record_audit` usa como ator padrão.
current_principal: ContextVar[Principal | None] = ContextVar("current_principal", default=None)


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    """SHA-256 da chave. As chaves são aleatórias (256 bits): dispensam sal e KDF lento."""
    return hashlib.sha256(key.encode()).hexdigest()


def same_key(provided: str, expected: str) -> bool:
    return hmac.compare_digest(provided.encode(), expected.encode())


def can(role: str, required: str) -> bool:
    if required not in ROLES:
        raise ValueError(f"Unknown role '{required}'. Use one of: {', '.join(ROLES)}")
    return ROLES.get(role, 0) >= ROLES[required]


def required_role(method: str, path: str) -> str:
    """Papel mínimo para um método e caminho, quando a rota não declara um papel próprio."""
    if method.upper() in READ_METHODS:
        return "reader"
    if method.upper() == "POST" and path.rstrip("/").endswith(READER_POST_SUFFIXES):
        return "reader"
    return "operator"
