from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import HTTPException, status


class ModelNotFoundError(LookupError):
    """Não há artefato registrado para o modelo pedido."""


class ArtifactIntegrityError(RuntimeError):
    """O artefato não confere com o checksum (ou a assinatura) do registry."""


class FeatureMismatchError(ValueError):
    """As features enviadas não são as que o modelo registrado espera."""


class CanaryConflictError(RuntimeError):
    """O canário pedido conflita com o estado atual (sem produção, já ativo, já decidido)."""


class GateFailedError(RuntimeError):
    """A promoção para produção foi barrada pelo release gate."""

    def __init__(self, gate: dict[str, Any]) -> None:
        super().__init__("Release gate failed")
        self.gate = gate


@contextmanager
def invalid_input_as_422() -> Iterator[None]:
    """Converte `ValueError` de validação de domínio em HTTP 422 (em vez de 500)."""
    try:
        yield
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
