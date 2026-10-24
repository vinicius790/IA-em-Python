"""Router do domínio `governance`: handlers das rotas versionadas ``/api/vN`` deste domínio."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...api_v4 import _record_or_404
from ...config import get_settings
from ...database import get_db
from ...db_models import ApiUser
from ...governance.audit import recent_audit, record_audit
from ...governance.evidence import build_evidence_pack
from ...governance.pii import redact_text, scan_text
from ...governance.rbac import Principal, generate_key, hash_key
from ...schemas import PIIScanRequest, UserCreate, UserRead
from ...security import require_role
from ...services import get_service

router = APIRouter()
settings = get_settings()
service = get_service()


@router.get("/api/v4/audit", tags=["governance"])
def audit_events(limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)) -> list[dict]:
    return recent_audit(db, limit)


@router.get("/api/v4/governance/gate/{model_name}", tags=["governance"])
def release_gate(model_name: str, version: int | None = Query(None, ge=1)) -> dict:
    """Avalia o release gate sobre as métricas registradas de uma versão do modelo."""
    record = _record_or_404(model_name, version)
    return {"model": record["alias"], "stage": record["stage"], "gate": service.release_gate(record)}


@router.post("/api/v6/security/pii-scan", tags=["security"])
def pii_scan(payload: PIIScanRequest) -> dict:
    return {"report": scan_text(payload.text), "redacted": redact_text(payload.text)}


@router.post("/api/v6/governance/evidence-pack", tags=["governance"])
def evidence_pack(model_name: str = Query(max_length=128), version: int = Query(ge=1)) -> dict:
    """Gera o pacote de evidências (registro + release gate + hash) de uma versão do registry."""
    record = service.registry.get(model_name, version)
    if record is None:
        raise HTTPException(404, "Model version not found in the registry")
    output = settings.artifact_dir / "governance" / f"evidence-{model_name}-v{version}.json"
    return build_evidence_pack(record, service.release_gate(record), output)


@router.get("/api/v8/users/me", tags=["users"])
def whoami(request: Request) -> dict:
    """Quem a chave enviada identifica e com qual papel."""
    principal: Principal = request.state.principal
    return {"name": principal.name, "role": principal.role}


@router.get("/api/v8/users", response_model=list[UserRead], tags=["users"])
def users(_: Principal = Depends(require_role("admin")), db: Session = Depends(get_db)):
    return list(db.scalars(select(ApiUser).order_by(ApiUser.id)).all())


@router.post("/api/v8/users", status_code=status.HTTP_201_CREATED, tags=["users"])
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


@router.delete("/api/v8/users/{user_id}", response_model=UserRead, tags=["users"])
def revoke_user(user_id: int, _: Principal = Depends(require_role("admin")), db: Session = Depends(get_db)):
    """Revoga a chave do usuário (o registro fica, para a auditoria continuar legível)."""
    user = db.get(ApiUser, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    user.active = False
    db.commit()
    record_audit(db, "user_revoked", user=user.name)
    return user
