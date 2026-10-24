"""Helpers da API v8 (handlers em app.api.routers): datasets enviados, comparação de modelos, deploy canário e usuários com papéis."""

from __future__ import annotations

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from .config import get_settings
from .db_models import CanaryDeployment
from .services import get_service

settings = get_settings()
service = get_service()


# ------------------------------------------------------------------ datasets


async def _csv_text(request: Request) -> str:
    """Corpo da requisição como texto CSV (UTF-8). O tamanho já foi limitado pelo middleware."""
    try:
        return (await request.body()).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "CSV must be UTF-8 encoded") from exc


# ----------------------------------------------------------------- inference


# ---------------------------------------------------------------- comparison


# -------------------------------------------------------------------- canary


def _canary_or_404(db: Session, canary_id: int) -> CanaryDeployment:
    canary = db.get(CanaryDeployment, canary_id)
    if canary is None:
        raise HTTPException(404, "Canary not found")
    return canary


# --------------------------------------------------------------------- users

VERSION = 8


def __getattr__(name: str):
    """Compatibilidade: ``router`` agora é montado a partir de ``app.api.routers``."""
    if name == "router":
        from .api import versioned_router

        return versioned_router(VERSION)
    raise AttributeError(name)
