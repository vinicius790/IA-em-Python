"""Deploy canário e roteamento de predições (fatia de ``MLService``, issue #9)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db_models import CanaryDeployment, PredictionLog
from ..deployment.canary import ARM_WINDOW, PROMOTE, ROLLBACK, CanaryRule, arm_stats, choose_arm, decide
from ..errors import CanaryConflictError, FeatureMismatchError, GateFailedError, ModelNotFoundError
from ..governance.audit import record_audit
from ._base import ServiceBase


class CanaryMixin(ServiceBase):
    def active_canary(self, db: Session, model_name: str) -> CanaryDeployment | None:
        return db.scalar(
            select(CanaryDeployment).where(
                CanaryDeployment.model_name == model_name, CanaryDeployment.status == "active"
            )
        )

    def start_canary(
        self,
        db: Session,
        model_name: str,
        candidate_version: int,
        traffic_percent: float,
        rule: CanaryRule,
        *,
        created_by: str = "api",
    ) -> CanaryDeployment:
        """Abre um canário entre a versão em produção (estável) e uma candidata."""
        stable = self.registry.production(model_name)
        if stable is None:
            raise CanaryConflictError(f"Model '{model_name}' has no production version to act as stable")
        candidate = self.registry.get(model_name, candidate_version)
        if candidate is None:
            raise KeyError(f"Model {model_name} version {candidate_version} not found")
        if candidate["version"] == stable["version"] or candidate.get("stage") == "archived":
            raise CanaryConflictError("Candidate must be a non-archived version different from production")
        if self.active_canary(db, model_name) is not None:
            raise CanaryConflictError(f"Model '{model_name}' already has an active canary")
        canary = CanaryDeployment(
            model_name=model_name,
            stable_version=stable["version"],
            candidate_version=candidate_version,
            traffic_percent=traffic_percent,
            status="active",
            rule=rule.as_dict(),
            baseline_prediction_id=db.scalar(select(func.max(PredictionLog.id))) or 0,
            created_by=created_by,
        )
        db.add(canary)
        try:
            db.commit()
        except IntegrityError as exc:  # outra abertura simultânea gravou antes (índice único parcial)
            db.rollback()
            raise CanaryConflictError(f"Model '{model_name}' already has an active canary") from exc
        db.refresh(canary)
        record_audit(
            db,
            "canary_started",
            canary_id=canary.id,
            model=model_name,
            stable=stable["version"],
            candidate=candidate_version,
            traffic_percent=traffic_percent,
        )
        return canary

    def _canary_ready(self, db: Session, canary: CanaryDeployment) -> bool:
        """A candidata já tem `min_requests` predições? Uma contagem limitada (no máximo
        `min_requests` ids do índice) em vez do relatório completo a cada predição.

        Antes disso a regra só pode decidir esperar; a partir disso ela sempre decide (promove
        ou reverte), então avaliar ao atingir o mínimo dá a mesma decisão de avaliar sempre.
        """
        needed = CanaryRule(**canary.rule).min_requests
        ids = (
            select(PredictionLog.id)
            .where(
                PredictionLog.model_name == canary.model_name,
                PredictionLog.model_version == canary.candidate_version,
                (PredictionLog.canary_id == canary.id) | (PredictionLog.canary_id.is_(None)),
                (PredictionLog.canary_arm == "candidate") | (PredictionLog.canary_arm.is_(None)),
                PredictionLog.id > canary.baseline_prediction_id,
            )
            .limit(needed)
            .subquery()
        )
        return (db.scalar(select(func.count()).select_from(ids)) or 0) >= needed

    def _arm_rows(self, db: Session, canary: CanaryDeployment, arm: str) -> list[dict[str, Any]]:
        version = canary.stable_version if arm == "stable" else canary.candidate_version
        query = (
            select(PredictionLog)
            .where(
                PredictionLog.model_name == canary.model_name,
                PredictionLog.model_version == version,
                (PredictionLog.canary_id == canary.id) | (PredictionLog.canary_id.is_(None)),
                (PredictionLog.canary_arm == arm) | (PredictionLog.canary_arm.is_(None)),
                PredictionLog.id > canary.baseline_prediction_id,
            )
            .order_by(PredictionLog.id.desc())
            .limit(ARM_WINDOW)
        )
        return [
            {
                "confidence": row.confidence,
                "latency_ms": row.latency_ms,
                "predicted_class": row.predicted_class,
                "actual_class": row.actual_class,
            }
            for row in db.scalars(query).all()
        ]

    def canary_report(self, db: Session, canary: CanaryDeployment) -> dict[str, Any]:
        """Estado do canário com as métricas separadas de cada braço e a decisão da regra."""
        stable = arm_stats(self._arm_rows(db, canary, "stable"))
        candidate = arm_stats(self._arm_rows(db, canary, "candidate"))
        return {
            "id": canary.id,
            "model_name": canary.model_name,
            "status": canary.status,
            "traffic_percent": canary.traffic_percent,
            "rule": canary.rule,
            "arms": {
                "stable": {"version": canary.stable_version, **stable},
                "candidate": {"version": canary.candidate_version, **candidate},
            },
            "evaluation": decide(stable, candidate, CanaryRule(**canary.rule)),
            "decision": canary.decision,
            "created_by": canary.created_by,
            "created_at": canary.created_at,
            "decided_at": canary.decided_at,
        }

    def _claim_canary(
        self, db: Session, canary: CanaryDeployment, status: str, decision: dict[str, Any]
    ) -> None:
        """Fecha o canário só se ele ainda estiver ativo (UPDATE condicional e atômico).

        Predições concorrentes, a reversão manual e a avaliação podem decidir ao mesmo
        tempo; só quem fechar primeiro mexe no registry, os demais recebem conflito.
        """
        claimed = db.execute(
            update(CanaryDeployment)
            .where(CanaryDeployment.id == canary.id, CanaryDeployment.status == "active")
            .values(status=status, decision=decision, decided_at=datetime.now(UTC))
            .execution_options(synchronize_session=False)
        ).rowcount
        db.commit()
        db.refresh(canary)
        if not claimed:
            raise CanaryConflictError(f"Canary {canary.id} is already {canary.status}")

    def _audit_canary(self, db: Session, canary: CanaryDeployment) -> None:
        decision = canary.decision or {}
        record_audit(
            db,
            f"canary_{canary.status}",
            canary_id=canary.id,
            model=canary.model_name,
            candidate=canary.candidate_version,
            reason=decision.get("reason"),
            automatic=decision.get("automatic", False),
        )

    def _archive_candidate(self, canary: CanaryDeployment) -> None:
        self.registry.promote(
            canary.model_name, canary.candidate_version, "archived", notes=f"canary {canary.id} rolled back"
        )

    def rollback_canary(self, db: Session, canary: CanaryDeployment, decision: dict[str, Any]) -> None:
        """Encerra o canário e arquiva a candidata; a estável volta a receber todo o tráfego."""
        self._claim_canary(db, canary, "rolled_back", decision)
        self._archive_candidate(canary)
        self._audit_canary(db, canary)

    def evaluate_canary(
        self, db: Session, canary: CanaryDeployment, *, automatic: bool = False
    ) -> dict[str, Any]:
        """Aplica a regra: promove a candidata, reverte, ou não faz nada (`wait`).

        A promoção passa pelo release gate como qualquer outra; se o gate reprovar, o
        canário é revertido.
        """
        if canary.status != "active":
            raise CanaryConflictError(f"Canary {canary.id} is already {canary.status}")
        report = self.canary_report(db, canary)
        decision = report["evaluation"] | {"automatic": automatic}
        if decision["decision"] == PROMOTE:
            self._claim_canary(db, canary, "promoted", decision)
            try:
                self.promote(
                    db,
                    canary.model_name,
                    canary.candidate_version,
                    "production",
                    notes=f"canary {canary.id} promoted",
                )
            except GateFailedError as exc:
                canary.status = "rolled_back"
                canary.decision = decision | {
                    "decision": ROLLBACK,
                    "reason": "release gate failed",
                    "gate": exc.gate,
                }
                db.commit()
                self._archive_candidate(canary)
            self._audit_canary(db, canary)
        elif decision["decision"] == ROLLBACK:
            self.rollback_canary(db, canary, decision)
        return self.canary_report(db, canary)

    def _check_width(self, artifact: dict[str, Any], record: dict[str, Any], features: list[float]) -> None:
        expected = artifact["features"]
        if len(features) != len(expected):
            raise FeatureMismatchError(
                f"Model '{record['alias']}' expects {len(expected)} features ({', '.join(expected)}); "
                f"got {len(features)}"
            )

    def feature_vector(self, model_name: str, features: dict[str, float]) -> list[float]:
        """Ordena as features nomeadas na ordem que o modelo servido espera."""
        expected = self.load(self.serving_record(model_name))["features"]
        missing = [name for name in expected if name not in features]
        unknown = sorted(set(features) - set(expected))
        if missing or unknown:
            raise FeatureMismatchError(
                f"Model '{model_name}' expects features {expected}; missing {missing}, unknown {unknown}"
            )
        return [float(features[name]) for name in expected]

    def predict_routed(
        self,
        db: Session,
        model_name: str,
        features: dict[str, float],
        *,
        request_id: str | None = None,
        routing_key: str | None = None,
    ) -> dict[str, Any]:
        """Predição com features nomeadas, para qualquer dataset, respeitando o canário ativo.

        Com canário ativo, `routing_key` (ou o id da requisição) decide o braço. Cada predição
        da candidata faz só uma contagem limitada; ao atingir `min_requests` a regra é aplicada
        e promove ou reverte sozinha.
        """
        vector = self.feature_vector(model_name, features)
        canary = self.active_canary(db, model_name)
        arm, records = "stable", None
        if canary is not None:
            arm = choose_arm(routing_key or request_id or "", canary.traffic_percent)
            version = canary.candidate_version if arm == "candidate" else canary.stable_version
            record = self.registry.get(model_name, version)
            if record is None:
                raise ModelNotFoundError(f"Canary {canary.id} points to missing version {version}")
            records = {model_name: record}
        result = self.predict_batch(
            db,
            [(model_name, vector)],
            request_id=request_id,
            records=records,
            canary_id=canary.id if canary else None,
            canary_arm=arm if canary else None,
        )[0]
        result: dict[str, Any] = dict(result)
        result["arm"] = arm
        result["canary_id"] = canary.id if canary else None
        if canary is not None and arm == "candidate" and self._canary_ready(db, canary):
            try:
                result["canary_status"] = self.evaluate_canary(db, canary, automatic=True)["status"]
            except CanaryConflictError:  # outra requisição decidiu o canário ao mesmo tempo
                db.refresh(canary)
                result["canary_status"] = canary.status
        elif canary is not None:  # estável, ou candidata ainda abaixo de `min_requests` (regra: esperar)
            result["canary_status"] = canary.status
        return result
