from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db_models import AuditEvent
from ..tracing import current_trace_id
from .rbac import current_principal


def record_audit(db: Session, action: str, *, actor: str | None = None, **payload: Any) -> None:
    """Grava um evento de auditoria na tabela `audit_events`.

    Sem `actor` explícito, o ator é o usuário autenticado da requisição (com o papel
    no payload) ou `api` fora de uma requisição (CLI, jobs). Com tracing ativo (extra
    `otel`), o `trace_id` do span corrente entra no payload.
    """
    principal = current_principal.get()
    if actor is None and principal is not None:
        actor, payload = principal.name, {**payload, "actor_role": principal.role}
    trace_id = current_trace_id()
    if trace_id is not None:
        payload = {**payload, "trace_id": trace_id}
    db.add(AuditEvent(action=action, actor=actor or "api", payload=payload))
    db.commit()


def recent_audit(db: Session, limit: int = 100) -> list[dict[str, Any]]:
    rows = db.scalars(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(limit)).all()
    return [
        {"id": r.id, "action": r.action, "actor": r.actor, "payload": r.payload, "created_at": r.created_at}
        for r in rows
    ]
