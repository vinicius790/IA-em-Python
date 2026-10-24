"""Routers HTTP por domínio, na ordem de montagem em ``app.main``."""

from . import data, governance, inference, labs, monitoring, registry, training

DOMAIN_ROUTERS = (data, training, registry, inference, monitoring, governance, labs)

__all__ = ["DOMAIN_ROUTERS"]
