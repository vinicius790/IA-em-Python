"""Camada HTTP organizada por domínio (issue #9).

Os endpoints continuam implementados nos módulos versionados ``app.api_v1`` ..
``app.api_v8`` (os paths públicos não mudam). Este pacote os reagrupa em
routers por domínio de negócio, que são o ponto único de montagem em
``app.main``. Cada rota versionada pertence a exatamente um domínio.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.routing import APIRoute

from .. import api_v1, api_v2, api_v3, api_v4, api_v5, api_v6, api_v7, api_v8

VERSIONED_ROUTERS: tuple[APIRouter, ...] = (
    api_v1.router,
    api_v2.router,
    api_v3.router,
    api_v4.router,
    api_v5.router,
    api_v6.router,
    api_v7.router,
    api_v8.router,
)

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


def build_domain_router(domain: str) -> APIRouter:
    """Router com as rotas versionadas do domínio, na ordem original de registro."""
    if domain not in DOMAINS:
        raise ValueError(f"Domínio desconhecido: {domain}")
    router = APIRouter()
    for versioned in VERSIONED_ROUTERS:
        for route in versioned.routes:
            if isinstance(route, APIRoute) and domain_of(route.path) == domain:
                router.routes.append(route)
    return router
