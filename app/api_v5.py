"""API v5 (Studio): fronteira de Pareto, busca na documentação e intervalo bootstrap."""

from __future__ import annotations

from fastapi import APIRouter

from .model_selection import pareto_frontier
from .modeling.evaluation import bootstrap_metric
from .schemas import EvaluatePredictionsRequest, SearchRequest
from .search.ingest import documentation_index
from .services import get_service

router = APIRouter(prefix="/api/v5")
service = get_service()


@router.get("/studio/frontier", tags=["studio"])
def frontier() -> list[dict]:
    """Versões registradas não dominadas em F1 macro (maior) x latência (menor)."""
    return pareto_frontier(service.list_models())


@router.post("/studio/search", tags=["studio"])
def search(payload: SearchRequest) -> list[dict]:
    """Busca TF-IDF no README e em `docs/` (não é busca semântica nem RAG)."""
    return [
        {"id": doc.id, "metadata": doc.metadata, "text": doc.text[:500]}
        for doc in documentation_index().search(payload.query, payload.k)
    ]


@router.post("/studio/evaluate-predictions", tags=["studio"])
def evaluate_predictions(payload: EvaluatePredictionsRequest) -> dict:
    """Estimativa e intervalo de confiança bootstrap (95%) para vetores de rótulos enviados."""
    return bootstrap_metric(payload.y_true, payload.y_pred, payload.metric, payload.iterations, payload.seed)
