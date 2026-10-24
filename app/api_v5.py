"""Helpers da API v5 (Studio; handlers em app.api.routers): fronteira de Pareto, busca na documentação e intervalo bootstrap."""

from __future__ import annotations

from .services import get_service

service = get_service()

VERSION = 5


def __getattr__(name: str):
    """Compatibilidade: ``router`` agora é montado a partir de ``app.api.routers``."""
    if name == "router":
        from .api import versioned_router

        return versioned_router(VERSION)
    raise AttributeError(name)
