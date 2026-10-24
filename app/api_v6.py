"""Helpers da API v6 (handlers em app.api.routers): pesquisa (matriz de experimentos), dados sintéticos, robustez, PII, SBOM e evidências."""

from __future__ import annotations

from .config import get_settings
from .services import get_service

settings = get_settings()
service = get_service()

VERSION = 6


def __getattr__(name: str):
    """Compatibilidade: ``router`` agora é montado a partir de ``app.api.routers``."""
    if name == "router":
        from .api import versioned_router

        return versioned_router(VERSION)
    raise AttributeError(name)
