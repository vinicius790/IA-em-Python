"""Helpers da API v3 (handlers em app.api.routers): metadados da plataforma, perfil e catálogo de datasets."""

from __future__ import annotations

from .config import get_settings
from .data import FEATURE_COLUMNS, TARGET_COLUMN
from .data_catalog.contracts import ColumnContract, DatasetContract

settings = get_settings()

IRIS_CONTRACT = DatasetContract(
    "iris",
    [
        *(
            ColumnContract(name, "float64", nullable=False, min_value=0, max_value=20)
            for name in FEATURE_COLUMNS
        ),
        ColumnContract(TARGET_COLUMN, nullable=False),
    ],
    min_rows=100,
    target=TARGET_COLUMN,
)

VERSION = 3


def __getattr__(name: str):
    """Compatibilidade: ``router`` agora é montado a partir de ``app.api.routers``."""
    if name == "router":
        from .api import versioned_router

        return versioned_router(VERSION)
    raise AttributeError(name)
