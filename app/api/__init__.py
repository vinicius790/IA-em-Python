"""Camada HTTP organizada por domínio (issue #9).

Os handlers ficam em ``app.api.routers.<domínio>``, com os paths públicos
versionados (``/api/vN/...``) inalterados. Os módulos ``app.api_v1`` ..
``app.api_v8`` guardam só helpers compartilhados e, por compatibilidade,
expõem ``router`` com as rotas da sua versão (ver ``versioned_router``).
Cada rota versionada pertence a exatamente um domínio.
"""

from __future__ import annotations

from importlib import import_module

from fastapi import APIRouter
from fastapi.routing import APIRoute

# Primeiro segmento após ``/api/vN`` -> domínio.
SEGMENT_DOMAINS: dict[str, str] = {
    "dataset": "data",
    "datasets": "data",
    "iris": "data",
    "data": "data",
    "features": "data",
    "training": "training",
    "pipelines": "training",
    "studio": "training",
    "research": "training",
    "modeling": "training",
    "models": "registry",
    "experiments": "registry",
    "comparisons": "registry",
    "deployment": "registry",
    "canary": "registry",
    "predict": "inference",
    "predictions": "inference",
    "explain": "inference",
    "monitoring": "monitoring",
    "jobs": "monitoring",
    "platform": "monitoring",
    "meta": "monitoring",
    "analytics": "monitoring",
    "audit": "governance",
    "governance": "governance",
    "security": "governance",
    "users": "governance",
    "labs": "labs",
}

DOMAINS: tuple[str, ...] = ("data", "training", "registry", "inference", "monitoring", "governance", "labs")


def domain_of(path: str) -> str:
    """Domínio de um path versionado, ex.: ``/api/v8/canary/{id}`` -> ``registry``."""
    parts = path.strip("/").split("/")
    if len(parts) < 3 or parts[0] != "api":
        raise ValueError(f"Path fora do padrão /api/vN/<segmento>: {path}")
    try:
        return SEGMENT_DOMAINS[parts[2]]
    except KeyError as exc:
        raise ValueError(f"Segmento sem domínio mapeado em {path}; atualize SEGMENT_DOMAINS") from exc


def _version_of(path: str) -> int:
    return int(path.split("/")[2].removeprefix("v"))


def build_domain_router(domain: str) -> APIRouter:
    """Router com as rotas versionadas do domínio (definidas em ``app.api.routers``)."""
    if domain not in DOMAINS:
        raise ValueError(f"Domínio desconhecido: {domain}")
    return import_module(f"{__name__}.routers.{domain}").router


def versioned_router(version: int) -> APIRouter:
    """Router de compatibilidade com as rotas de ``/api/v{version}``, em ordem de domínio."""
    router = APIRouter()
    for domain in DOMAINS:
        for route in build_domain_router(domain).routes:
            if isinstance(route, APIRoute) and _version_of(route.path) == version:
                router.routes.append(route)
    return router


VERSIONED_ROUTERS: tuple[APIRouter, ...] = tuple(versioned_router(version) for version in range(1, 9))
