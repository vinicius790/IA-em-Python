"""API v8: datasets enviados, comparação de modelos, deploy canário e usuários com papéis."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import get_settings
from .data_catalog.catalog import public_record
from .data_catalog.csv_profile import profile_csv, training_problems
from .database import get_db
from .db_models import ApiUser, CanaryDeployment
from .deployment.canary import CanaryRule
from .errors import CanaryConflictError, invalid_input_as_422
from .governance.audit import record_audit
from .governance.rbac import Principal, generate_key, hash_key
from .schemas import (
    CanaryRequest,
    CanaryTrafficRequest,
    ComparisonRequest,
    NamedExplainRequest,
    NamedPredictionRequest,
    UserCreate,
    UserRead,
)
from .security import require_role
from .services import get_service

router = APIRouter(prefix="/api/v8")
settings = get_settings()
service = get_service()


# ------------------------------------------------------------------ datasets


async def _csv_text(request: Request) -> str:
    """Corpo da requisição como texto CSV (UTF-8). O tamanho já foi limitado pelo middleware."""
    try:
        return (await request.body()).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "CSV must be UTF-8 encoded") from exc


@router.post("/datasets/profile", tags=["datasets"])
async def dataset_profile(request: Request, target: str | None = Query(None, max_length=128)) -> dict:
    """Perfil de um CSV enviado no corpo (`Content-Type: text/csv`), sem gravar nada."""
    text = await _csv_text(request)
    with invalid_input_as_422():
        profile = await run_in_threadpool(profile_csv, text, target)
    return profile | {"training_problems": training_problems(profile)}


@router.post("/datasets/upload", status_code=status.HTTP_201_CREATED, tags=["datasets"])
async def dataset_upload(
    request: Request,
    name: str = Query(min_length=1, max_length=64),
    target: str | None = Query(None, max_length=128),
    db: Session = Depends(get_db),
) -> dict:
    """Recebe um CSV no corpo (`Content-Type: text/csv`), faz o perfil e registra no catálogo.

    Sem `target`, o alvo é inferido. Reenviar o mesmo `name` registra uma nova versão.
    """
    text = await _csv_text(request)
    with invalid_input_as_422():
        return public_record(await run_in_threadpool(service.register_dataset, db, name, text, target))


@router.get("/datasets", tags=["datasets"])
def datasets() -> list[dict]:
    """Datasets enviados (a versão mais recente de cada nome), sem o perfil completo."""
    latest = {record["name"]: record for record in service.catalog().list() if record.get("target")}
    return [
        {key: value for key, value in public_record(record).items() if key != "profile"}
        for record in latest.values()
    ]


@router.get("/datasets/{name}", tags=["datasets"])
def dataset(name: str) -> dict:
    record = service.catalog().get(name)
    if record is None or not record.get("target"):
        raise HTTPException(404, "Dataset not found")
    return public_record(record)


# ----------------------------------------------------------------- inference


@router.post("/predict", tags=["inference"])
def predict(payload: NamedPredictionRequest, request: Request, db: Session = Depends(get_db)) -> dict:
    """Predição com features nomeadas, para modelos de qualquer dataset.

    Se o modelo tem um canário ativo, o tráfego é dividido entre a versão estável e a
    candidata; a resposta diz qual braço (`arm`) atendeu.
    """
    return service.predict_routed(
        db,
        payload.model_name,
        payload.features,
        request_id=request.state.request_id,
        routing_key=payload.routing_key,
    )


@router.post("/explain", tags=["explainability"])
def explain(payload: NamedExplainRequest) -> dict:
    """Shapley exato com features nomeadas (até 10 features)."""
    with invalid_input_as_422():
        vector = service.feature_vector(payload.model_name, payload.features)
        return service.explain(payload.model_name, vector)


# ---------------------------------------------------------------- comparison


@router.post("/comparisons", status_code=status.HTTP_201_CREATED, tags=["comparison"])
def create_comparison(payload: ComparisonRequest, db: Session = Depends(get_db)) -> dict:
    """Validação cruzada estratificada dos modelos pedidos, nas mesmas dobras, com intervalo
    de confiança t; grava um relatório JSON e um HTML por execução."""
    with invalid_input_as_422():
        return service.compare(
            db,
            payload.models,
            dataset=payload.dataset,
            folds=payload.folds,
            confidence=payload.confidence,
            random_state=payload.random_state,
        )


@router.get("/comparisons", tags=["comparison"])
def comparisons(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    return service.list_comparisons(limit)


@router.get("/comparisons/{run_id}", tags=["comparison"])
def comparison(run_id: str) -> dict:
    path = service.comparison_file(run_id, "json")
    if path is None:
        raise HTTPException(404, "Comparison run not found")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/comparisons/{run_id}/html", response_class=HTMLResponse, tags=["comparison"])
def comparison_html(run_id: str) -> HTMLResponse:
    path = service.comparison_file(run_id, "html")
    if path is None:
        raise HTTPException(404, "Comparison run not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


# -------------------------------------------------------------------- canary


def _canary_or_404(db: Session, canary_id: int) -> CanaryDeployment:
    canary = db.get(CanaryDeployment, canary_id)
    if canary is None:
        raise HTTPException(404, "Canary not found")
    return canary


@router.post("/canary", status_code=status.HTTP_201_CREATED, tags=["canary"])
def start_canary(payload: CanaryRequest, request: Request, db: Session = Depends(get_db)) -> dict:
    """Abre um canário: `traffic_percent`% das predições de `/api/v8/predict` vão para a candidata."""
    principal: Principal = request.state.principal
    try:
        with invalid_input_as_422():
            rule = CanaryRule(**payload.rule.model_dump())
        canary = service.start_canary(
            db,
            payload.model_name,
            payload.candidate_version,
            payload.traffic_percent,
            rule,
            created_by=principal.name,
        )
    except KeyError as exc:
        raise HTTPException(404, "Candidate version not found in the registry") from exc
    except CanaryConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return service.canary_report(db, canary)


@router.get("/canary", tags=["canary"])
def canaries(
    model_name: str | None = Query(None, max_length=128),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[dict]:
    query = select(CanaryDeployment).order_by(CanaryDeployment.id.desc()).limit(limit)
    if model_name:
        query = query.where(CanaryDeployment.model_name == model_name)
    return [
        {
            "id": row.id,
            "model_name": row.model_name,
            "status": row.status,
            "stable_version": row.stable_version,
            "candidate_version": row.candidate_version,
            "traffic_percent": row.traffic_percent,
            "created_at": row.created_at,
            "decided_at": row.decided_at,
        }
        for row in db.scalars(query).all()
    ]


@router.get("/canary/{canary_id}", tags=["canary"])
def canary_status(canary_id: int, db: Session = Depends(get_db)) -> dict:
    """Métricas separadas da estável e da candidata e o que a regra decidiria agora."""
    return service.canary_report(db, _canary_or_404(db, canary_id))


@router.post("/canary/{canary_id}/evaluate", tags=["canary"])
def evaluate_canary(canary_id: int, db: Session = Depends(get_db)) -> dict:
    """Aplica a regra agora: promove, reverte ou mantém o canário."""
    try:
        return service.evaluate_canary(db, _canary_or_404(db, canary_id))
    except CanaryConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/canary/{canary_id}/rollback", tags=["canary"])
def rollback_canary(canary_id: int, db: Session = Depends(get_db)) -> dict:
    """Reversão manual: encerra o canário e arquiva a candidata."""
    canary = _canary_or_404(db, canary_id)
    try:
        service.rollback_canary(db, canary, {"decision": "rollback", "reason": "manual rollback"})
    except CanaryConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return service.canary_report(db, canary)


@router.post("/canary/{canary_id}/traffic", tags=["canary"])
def canary_traffic(canary_id: int, payload: CanaryTrafficRequest, db: Session = Depends(get_db)) -> dict:
    """Muda o percentual de tráfego da candidata de um canário ativo."""
    canary = _canary_or_404(db, canary_id)
    if canary.status != "active":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Canary {canary.id} is already {canary.status}")
    canary.traffic_percent = payload.traffic_percent
    db.commit()
    record_audit(db, "canary_traffic", canary_id=canary.id, traffic_percent=payload.traffic_percent)
    return service.canary_report(db, canary)


# --------------------------------------------------------------------- users


@router.get("/users/me", tags=["users"])
def whoami(request: Request) -> dict:
    """Quem a chave enviada identifica e com qual papel."""
    principal: Principal = request.state.principal
    return {"name": principal.name, "role": principal.role}


@router.get("/users", response_model=list[UserRead], tags=["users"])
def users(_: Principal = Depends(require_role("admin")), db: Session = Depends(get_db)):
    return list(db.scalars(select(ApiUser).order_by(ApiUser.id)).all())


@router.post("/users", status_code=status.HTTP_201_CREATED, tags=["users"])
def create_user(
    payload: UserCreate, _: Principal = Depends(require_role("admin")), db: Session = Depends(get_db)
) -> dict:
    """Cria um usuário e devolve a chave dele. A chave só aparece nesta resposta (o banco guarda o hash)."""
    key = generate_key()
    user = ApiUser(name=payload.name, role=payload.role, key_hash=hash_key(key))
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, f"User '{payload.name}' already exists") from exc
    record_audit(db, "user_created", user=user.name, role=user.role)
    return {"id": user.id, "name": user.name, "role": user.role, "active": True, "api_key": key}


@router.delete("/users/{user_id}", response_model=UserRead, tags=["users"])
def revoke_user(user_id: int, _: Principal = Depends(require_role("admin")), db: Session = Depends(get_db)):
    """Revoga a chave do usuário (o registro fica, para a auditoria continuar legível)."""
    user = db.get(ApiUser, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    user.active = False
    db.commit()
    record_audit(db, "user_revoked", user=user.name)
    return user
