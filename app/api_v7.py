"""Helpers da API v7 (Labs; handlers em app.api.routers): AutoML tabular, classificação de texto, previsão de séries e grafo de linhagem."""

from __future__ import annotations

from .services import get_service

service = get_service()

MAX_BACKTEST_HORIZON = 3

VERSION = 7


def __getattr__(name: str):
    """Compatibilidade: ``router`` agora é montado a partir de ``app.api.routers``."""
    if name == "router":
        from .api import versioned_router

        return versioned_router(VERSION)
    raise AttributeError(name)
