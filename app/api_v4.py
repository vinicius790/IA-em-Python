"""Helpers da API v4 (handlers em app.api.routers): pipelines, auditoria, governança e análises do dataset."""

from __future__ import annotations

from fastapi import HTTPException

from .config import get_settings
from .services import get_service

settings = get_settings()
service = get_service()


def _record_or_404(model_name: str, version: int | None) -> dict:
    record = service.registry.get(model_name, version) if version else service.latest_model(model_name)
    if record is None:
        raise HTTPException(404, "Model version not found in the registry")
    return record


VERSION = 4


def __getattr__(name: str):
    """Compatibilidade: ``router`` agora é montado a partir de ``app.api.routers``."""
    if name == "router":
        from .api import versioned_router

        return versioned_router(VERSION)
    raise AttributeError(name)
