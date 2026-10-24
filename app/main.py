"""Iris AI Platform: aplicação FastAPI principal com MLOps, Governança Ética e XAI."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

from . import __version__, api_v1
from .api.routers import DOMAIN_ROUTERS
from .config import get_settings
from .data import FEATURE_COLUMNS, load_iris_dataframe
from .database import database_ok, engine, init_db
from .errors import (
    ArtifactIntegrityError,
    CanaryConflictError,
    FeatureMismatchError,
    ModelNotFoundError,
)
from .features.feature_store import get_feature_store
from .governance.fairness import audit_fairness
from .modeling.explainability import compute_kernel_shap
from .monitoring.drift import detect_drift
from .observability import configure_logging, get_logger, record_request
from .security import (
    BodySizeLimitMiddleware,
    RateLimiter,
    require_api_key,
    sanitize_request_id,
)
from .services import get_service
from .tracing import setup_tracing

settings = get_settings()
configure_logging(level="DEBUG" if settings.debug else "INFO")
logger = get_logger()

# Limitador de taxa global configurado pelas variáveis de ambiente
rate_limiter = RateLimiter(
    limit=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Aplica as migrações Alembic antes de servir: a API sobe com o schema completo."""
    init_db()
    yield


app = FastAPI(
    lifespan=lifespan,
    title=settings.app_name,
    version=__version__,
    description=(
        "Plataforma Corporativa de Machine Learning, MLOps, Governança Ética (Fairness & Bias) "
        "e Explicabilidade Avançada de Modelos (XAI)."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)
# OpenTelemetry (extra `otel`): no-op sem o extra; exporta OTLP só com endpoint configurado.
setup_tracing(app, engine)

# ------------------------------------------------------------------ Middlewares


def _find_route_path(routes: Any, scope: dict) -> str | None:
    """Busca recursivamente o template de rota, inclusive em _IncludedRouter e subrouters."""
    for r in routes:
        if hasattr(r, "original_router") and hasattr(r.original_router, "routes"):
            matched = _find_route_path(r.original_router.routes, scope)
            if matched:
                return matched
        if hasattr(r, "routes"):
            matched = _find_route_path(r.routes, scope)
            if matched:
                return matched
        if hasattr(r, "matches"):
            match, _ = r.matches(scope)
            if match == Match.FULL:
                return getattr(r, "path", None)
    return None


class ObservabilityAndSecurityMiddleware(BaseHTTPMiddleware):
    """Middleware para Request-ID, Rate Limiting e Métricas Prometheus por template de rota."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = perf_counter()
        raw_request_id = request.headers.get("x-request-id")
        request_id = sanitize_request_id(raw_request_id)
        request.state.request_id = request_id

        # 1. Rate limiting por cliente
        client_key = request.headers.get("x-api-key") or (
            request.client.host if request.client else "unknown"
        )
        if not rate_limiter.allow(client_key):
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(rate_limiter.window_seconds), "X-Request-ID": request_id},
            )
            return response

        # 2. Execução da rota
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception("unhandled_error", exc_info=exc, request_id=request_id)
            route_path = _find_route_path(app.routes, request.scope)
            record_request(start_time, request.method, route_path, 500)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )

        response.headers["X-Request-ID"] = request_id

        # 3. Descobrir o template da rota para cardinalidade controlada nas métricas Prometheus
        route_path = _find_route_path(app.routes, request.scope)
        record_request(start_time, request.method, route_path, response.status_code)
        return response


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or sanitize_request_id(
        request.headers.get("x-request-id")
    )
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc), "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(ArtifactIntegrityError)
async def artifact_integrity_handler(request: Request, exc: ArtifactIntegrityError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or sanitize_request_id(
        request.headers.get("x-request-id")
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(FeatureMismatchError)
async def feature_mismatch_handler(request: Request, exc: FeatureMismatchError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or sanitize_request_id(
        request.headers.get("x-request-id")
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(exc), "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(CanaryConflictError)
async def canary_conflict_handler(request: Request, exc: CanaryConflictError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or sanitize_request_id(
        request.headers.get("x-request-id")
    )
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": str(exc), "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


def _sanitize_err_value(val: Any) -> Any:
    if isinstance(val, dict):
        return {k: _sanitize_err_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_err_value(v) for v in val]
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        return val
    if isinstance(val, Exception):
        return str(val)
    return val


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or sanitize_request_id(
        request.headers.get("x-request-id")
    )
    sanitized_errors = jsonable_encoder(_sanitize_err_value(exc.errors()))
    headers = {"X-Request-ID": request_id} if request_id else {}
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": sanitized_errors},
        headers=headers,
    )


# Tratamento seguro de exceções não tratadas (HTTP 500)
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None) or {},
        )
    request_id = getattr(request.state, "request_id", None) or sanitize_request_id(
        request.headers.get("x-request-id")
    )
    logger.exception("unhandled_error", exc_info=exc, request_id=request_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


# Registrar middlewares
app.add_middleware(ObservabilityAndSecurityMiddleware)
app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ------------------------------------------------------------------ Sondas Públicas e Console


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/healthz", tags=["system"])
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/readyz", tags=["system"])
def readyz() -> dict[str, Any]:
    db_ok = database_ok()
    return {"status": "ready" if db_ok else "not_ready", "database": db_ok}


console_dir = Path(__file__).parent / "console"
if console_dir.is_dir():
    app.mount("/console", StaticFiles(directory=console_dir, html=True), name="console")

# ------------------------------------------------------------------ Inclusão dos Routers por Domínio
app.include_router(api_v1.public_router)
for _domain in DOMAIN_ROUTERS:
    app.include_router(_domain.router, dependencies=[Depends(require_api_key)])


# ------------------------------------------------------------------ Novos Schemas e Endpoints MLOps/Governança


class OnlineLookupRequest(BaseModel):
    view_name: str = Field(default="iris_measurements", description="Nome da Feature View cadastrada")
    entity_keys: list[Any] = Field(..., description="Chaves primárias de identificação para lookup")
    features: list[str] | None = Field(default=None, description="Features desejadas (None para todas)")
    version: str | None = Field(default=None, description="Versão semântica da view (ex: 'v1')")


class FairnessAuditRequest(BaseModel):
    y_true: list[Any] = Field(..., description="Valores reais do rótulo/alvo")
    y_pred: list[Any] = Field(..., description="Predições emitidas pelo modelo avaliado")
    protected_attribute: list[Any] = Field(
        ..., description="Valores do atributo protegido (ex: gênero, etnia, idade)"
    )
    positive_label: Any = Field(default=1, description="Valor que denota o desfecho favorável/positivo")
    privileged_group: Any | None = Field(
        default=None, description="Identificador do grupo privilegiado de referência"
    )
    disparate_impact_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Limiar legal do Impacto Disparate"
    )
    equal_opportunity_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Limiar aceitável da Igualdade de Oportunidade"
    )
    attribute_name: str = Field(
        default="protected_attribute", description="Nome semântico do atributo protegido auditado"
    )


class ExplainRequest(BaseModel):
    instance: dict[str, float] = Field(
        ...,
        description="Medidas numéricas da instância para explicar localmente",
        json_schema_extra={
            "example": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        },
    )
    model_name: str = Field(
        default="logistic_regression", description="Modelo registrado no catálogo a utilizar"
    )
    target_class: int | None = Field(
        default=0, description="Índice da classe alvo para decomposição de probabilidades"
    )
    n_samples: int = Field(default=128, ge=16, le=1024, description="Número de amostras para o Kernel SHAP")


@app.post("/features/online-lookup", tags=["features"], dependencies=[Depends(require_api_key)])
def features_online_lookup(request: OnlineLookupRequest) -> dict[str, Any]:
    """Recuperação de atributos em baixa latência (sub-milissegundo) via Feature Store Corporativa."""
    store = get_feature_store()
    try:
        results = store.lookup_online(
            view_name=request.view_name,
            entity_keys=request.entity_keys,
            features=request.features,
            version=request.version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return {
        "view_name": request.view_name,
        "results": results,
        "count": len(results),
        "hit_rate": (round(store.online_store.stats["hits"] / max(1, store.online_store.stats["reads"]), 4)),
    }


@app.post("/governance/fairness-audit", tags=["governance"], dependencies=[Depends(require_api_key)])
def governance_fairness_audit(request: FairnessAuditRequest) -> dict[str, Any]:
    """Auditoria ética e avaliação de viés algorítmico em atributos protegidos com mitigação."""
    try:
        report = audit_fairness(
            y_true=request.y_true,
            y_pred=request.y_pred,
            protected_attribute=request.protected_attribute,
            positive_label=request.positive_label,
            privileged_group=request.privileged_group,
            disparate_impact_threshold=request.disparate_impact_threshold,
            equal_opportunity_threshold=request.equal_opportunity_threshold,
            attribute_name=request.attribute_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc

    return report.to_dict()


@app.post("/model/explain", tags=["modeling"], dependencies=[Depends(require_api_key)])
def model_explain(request: ExplainRequest) -> dict[str, Any]:
    """Decomposição local de predições via Kernel SHAP simplificado com atribuição aditiva de features."""
    service = get_service()
    model_record = service.latest_model(request.model_name)
    if model_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo '{request.model_name}' não encontrado no registry.",
        )

    # Carregar modelo salvo
    loaded = service.load(model_record)["model"]
    background_df = load_iris_dataframe()[FEATURE_COLUMNS]

    try:
        decomp = compute_kernel_shap(
            model=loaded,
            background=background_df,
            instance=request.instance,
            feature_names=FEATURE_COLUMNS,
            target_class=request.target_class,
            n_samples=request.n_samples,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Falha na decomposição SHAP: {exc}",
        ) from exc

    return decomp.to_dict()


# ------------------------------------------------------------------ Retrocompatibilidade


class LegacyBatchRequest(BaseModel):
    data: list[dict]
    reference_data: list[dict] | None = None


@app.post("/predict/batch", include_in_schema=False)
def legacy_batch_predict(request: LegacyBatchRequest) -> dict[str, Any]:
    """Endpoint legado de inferência em lote com análise de drift."""
    if not request.data:
        raise HTTPException(status_code=400, detail="Empty data list")

    df = pd.DataFrame(request.data)
    predictions = [row.get("value", 0) * 1.05 for row in request.data]

    drift_info = None
    if request.reference_data:
        ref_df = pd.DataFrame(request.reference_data)
        drift_info = detect_drift(ref_df, df)

    return {
        "predictions": predictions,
        "count": len(predictions),
        "drift_analysis": drift_info,
    }
