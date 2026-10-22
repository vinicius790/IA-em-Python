from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .data import FEATURE_COLUMNS, TARGET_COLUMN, load_iris_dataframe
from .data_catalog.catalog import DatasetCatalog
from .data_catalog.csv_profile import (
    is_null,
    profile_csv,
    read_rows,
    training_problems,
    valid_dataset_name,
)
from .db_models import (
    CanaryDeployment,
    ExperimentRun,
    IrisRecord,
    MonitoringReport,
    PredictionLog,
    TrainingRun,
)
from .deployment.canary import (
    ARM_WINDOW,
    PROMOTE,
    ROLLBACK,
    CanaryRule,
    arm_stats,
    choose_arm,
    decide,
    validate_canary_attribution,
)
from .deployment.gates import evaluate_release_gate
from .errors import (
    ArtifactIntegrityError,
    CanaryConflictError,
    FeatureMismatchError,
    GateFailedError,
    ModelNotFoundError,
)
from .experiment_tracking import ExperimentTracker
from .explainability import shapley_explanation
from .governance.audit import record_audit
from .ml import load_artifact, predict_artifact, train_model
from .modeling.comparison import compare_models
from .monitoring.drift import drift_report
from .monitoring.health import score_health
from .monitoring.sla import SLO, SLOEvaluator
from .observability import (
    DRIFT_COUNT,
    DRIFT_PSI,
    HEALTH_SCORE,
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    TRAINING_COUNT,
    get_logger,
)
from .platform.cache import prediction_cache
from .platform.hash_utils import canonical_hash
from .quality import validate_training_frame
from .registry import ModelRegistry
from .reports.generator import comparison_report_html

logger = get_logger()

MONITORING_WINDOW = 500
MIN_DRIFT_SAMPLES = 30
IRIS_DATASET = "iris"
COMPARISON_RUN_ID = re.compile(r"\d{8}T\d{6}-[0-9a-f]{8}")


class MLService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.registry = ModelRegistry(settings.model_dir, settings.artifact_signing_key)
        self.tracker = ExperimentTracker(settings.mlflow_tracking_uri)
        # ponytail: cache sem limite, uma entrada por arquivo de artefato já servido.
        self._artifacts: dict[str, tuple[tuple[str, int], dict[str, Any]]] = {}

    # ------------------------------------------------------------------ registry

    def list_models(self) -> list[dict]:
        return self.registry.list()

    def latest_model(self, name: str) -> dict | None:
        return self.registry.latest(name)

    def serving_record(self, model_name: str) -> dict[str, Any]:
        """Versão que atende inferência: a de produção ou, na falta dela, a mais recente."""
        record = self.registry.production(model_name) or self.registry.latest(model_name)
        if not record:
            raise ModelNotFoundError(f"No registered artifact for model '{model_name}'. Train it first.")
        return record

    def load(self, record: dict[str, Any]) -> dict[str, Any]:
        """Único ponto que desserializa artefatos: sempre depois de verificar a integridade."""
        path = self.registry.artifact_file(record)
        stamp = (record["checksum_sha256"], path.stat().st_mtime_ns if path.is_file() else -1)
        cached = self._artifacts.get(str(path))
        if cached and cached[0] == stamp:
            return cached[1]
        if not self.registry.verify(record):
            raise ArtifactIntegrityError(f"Artifact integrity verification failed for '{record['alias']}'")
        artifact = load_artifact(path)
        self._artifacts[str(path)] = (stamp, artifact)
        return artifact

    def release_gate(self, record: dict[str, Any]) -> dict[str, Any]:
        return evaluate_release_gate(
            record.get("metrics") or {},
            min_accuracy=self.settings.gate_min_accuracy,
            min_f1=self.settings.gate_min_f1,
            max_latency_ms=self.settings.gate_max_latency_ms,
        )

    def promote(
        self, db: Session, model_name: str, version: int, stage: str, *, notes: str = "", force: bool = False
    ) -> dict[str, Any]:
        """Muda o estágio; ir para `production` exige passar no release gate (ou `force`)."""
        record = self.registry.get(model_name, version)
        if record is None:
            raise KeyError(f"Model {model_name} version {version} not found")
        gate = self.release_gate(record) if stage == "production" else None
        if gate and not gate["passed"] and not force:
            record_audit(db, "promotion_blocked", model=model_name, version=version, stage=stage, gate=gate)
            raise GateFailedError(gate)
        promoted = self.registry.promote(model_name, version, stage, notes=notes)
        record_audit(db, "promotion", model=model_name, version=version, stage=stage, forced=force, gate=gate)
        return promoted

    # ------------------------------------------------------------------ datasets

    def catalog(self) -> DatasetCatalog:
        return DatasetCatalog(self.settings.data_dir)

    def register_dataset(
        self, db: Session, name: str, text: str, target: str | None = None
    ) -> dict[str, Any]:
        """Valida, grava e registra no catálogo um CSV enviado. `ValueError` para entrada inválida."""
        if not valid_dataset_name(name) or name == IRIS_DATASET:
            raise ValueError(
                "Dataset name must match [a-z0-9][a-z0-9_-]{0,63} and must not be the reserved name 'iris'"
            )
        text = text.lstrip("\ufeff")
        profile = profile_csv(text, target)
        problems = training_problems(profile)
        if problems:
            raise ValueError(f"Dataset rejected: {'; '.join(problems)}")
        # Um arquivo por conte\u00fado: reenviar o nome cria outra vers\u00e3o sem apagar a anterior,
        # que continua sendo a refer\u00eancia dos modelos treinados nela.
        path = self._upload_path(name, hashlib.sha256(text.encode("utf-8")).hexdigest())
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="")
        record = self.catalog().register(name, path, target=profile["target"], features=profile["features"])
        record_audit(
            db,
            "dataset_upload",
            dataset=name,
            sha256=record["sha256"],
            rows=profile["rows"],
            target=profile["target"],
        )
        return record | {"classes": profile["classes"], "target_inferred": profile["target_inferred"]}

    def _upload_path(self, name: str, sha256: str) -> Path:
        """Arquivo de uma versão de dataset enviado (nome já validado, hash hexadecimal)."""
        return self.settings.data_dir / "uploads" / name / f"{sha256[:16]}.csv"

    def dataset_record(self, dataset: str, sha256: str | None = None) -> dict[str, Any]:
        """Registro do catálogo de um dataset enviado: a versão `sha256` ou a mais recente."""
        records = [
            record
            for record in (self.catalog().list(dataset) if valid_dataset_name(dataset) else [])
            if record.get("target") and sha256 in (None, record["sha256"])
        ]
        if not records or not self._upload_path(dataset, records[-1]["sha256"]).is_file():
            version = f" (version {sha256[:12]})" if sha256 else ""
            raise ValueError(
                f"Unknown dataset '{dataset}'{version}. Upload it first (POST /api/v8/datasets/upload)."
            )
        return records[-1]

    def uploaded_frame(self, record: dict[str, Any]) -> tuple[pd.DataFrame, list[str], str]:
        """DataFrame de um dataset enviado, lido pelo mesmo leitor e com as mesmas regras de
        nulo e de número do perfil que o aceitou (o `read_csv` do pandas diverge em ambos)."""
        path = self._upload_path(record["name"], record["sha256"])
        header, rows = read_rows(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(rows, columns=header).map(
            lambda value: None if is_null(value) else value.strip()
        )
        features = list(record["features"])
        frame[features] = frame[features].astype(float)
        return frame, features, record["target"]

    def dataset_frame(
        self, dataset: str | None, sha256: str | None = None
    ) -> tuple[pd.DataFrame, list[str] | None, str | None]:
        """DataFrame, features e alvo de um dataset do catálogo (a versão `sha256` ou a mais
        recente); `None`/`iris` devolve o Iris com `features` e alvo nulos (os padrões do treino)."""
        if dataset in (None, IRIS_DATASET):
            return load_iris_dataframe(), None, None
        return self.uploaded_frame(self.dataset_record(dataset, sha256))

    # ---------------------------------------------------------------- comparison

    def compare(
        self,
        db: Session,
        models: list[str],
        *,
        dataset: str | None = None,
        folds: int = 5,
        confidence: float = 0.95,
        random_state: int | None = None,
    ) -> dict[str, Any]:
        """Compara modelos por validação cruzada estratificada e grava o relatório JSON e HTML."""
        frame, features, target = self.dataset_frame(dataset)
        report = compare_models(
            frame,
            features or FEATURE_COLUMNS,
            target or TARGET_COLUMN,
            models,
            folds=folds,
            random_state=self.settings.random_state if random_state is None else random_state,
            confidence=confidence,
        )
        created_at = datetime.now(UTC)
        report["dataset"] = dataset or IRIS_DATASET
        report["created_at"] = created_at.isoformat()
        report["run_id"] = f"{created_at:%Y%m%dT%H%M%S}-{canonical_hash(report)[:8]}"
        directory = self._comparison_dir()
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / f"{report['run_id']}.json"
        html_path = comparison_report_html(report, directory / f"{report['run_id']}.html")
        report["reports"] = {"json": str(json_path), "html": str(html_path)}
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        record_audit(
            db,
            "model_comparison",
            run_id=report["run_id"],
            dataset=report["dataset"],
            models=models,
            best_model=report["best_model"],
        )
        return report

    def _comparison_dir(self) -> Path:
        return Path(self.settings.artifact_dir) / "comparisons"

    def list_comparisons(self, limit: int = 50) -> list[dict[str, Any]]:
        files = sorted(self._comparison_dir().glob("*.json"), reverse=True)[:limit]
        rows = []
        for file in files:
            data = json.loads(file.read_text(encoding="utf-8"))
            rows.append(
                {
                    "run_id": data["run_id"],
                    "dataset": data["dataset"],
                    "created_at": data["created_at"],
                    "best_model": data["best_model"],
                    "models": [row["model"] for row in data["results"]],
                }
            )
        return rows

    def comparison_file(self, run_id: str, suffix: str) -> Path | None:
        """Caminho do relatório de uma execução; o `run_id` é validado antes de virar caminho."""
        if not COMPARISON_RUN_ID.fullmatch(run_id):
            return None
        path = self._comparison_dir() / f"{run_id}.{suffix}"
        return path if path.is_file() else None

    # -------------------------------------------------------------------- canary

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

    # ------------------------------------------------------------------ training

    def train(
        self,
        db: Session,
        model_name: str,
        test_size: float = 0.2,
        cv_folds: int = 5,
        tune_hyperparameters: bool = False,
        optimization_trials: int = 12,
        dataset: str | None = None,
    ) -> TrainingRun:
        """Treina e registra `model_name`. Com `dataset` (nome no catálogo), o modelo é
        registrado como `<dataset>.<modelo>`; sem ele, o treino é o do Iris."""
        dataset = None if dataset in (None, IRIS_DATASET) else dataset
        algorithm = model_name
        model_name = f"{dataset}.{algorithm}" if dataset else algorithm
        parameters = {
            "test_size": test_size,
            "cv_folds": cv_folds,
            "tune_hyperparameters": tune_hyperparameters,
            "optimization_trials": optimization_trials,
            "random_state": self.settings.random_state,
            "dataset": dataset or IRIS_DATASET,
        }
        run = TrainingRun(model_name=model_name, status="running", parameters=parameters)
        db.add(run)
        db.commit()
        db.refresh(run)
        start = perf_counter()
        try:
            # A versão exata do dataset fica no registry: o monitoramento compara contra ela,
            # mesmo depois de um novo envio com o mesmo nome.
            source = self.dataset_record(dataset) if dataset else None
            frame, features, target = self.uploaded_frame(source) if source else self.dataset_frame(None)
            result = train_model(
                frame,
                algorithm,
                self.settings.model_dir,
                random_state=self.settings.random_state,
                test_size=test_size,
                cv_folds=cv_folds,
                tune_hyperparameters=tune_hyperparameters,
                optimization_trials=optimization_trials,
                features=features,
                target=target,
                dataset=dataset,
            )
            registration = self.registry.register(
                model_name,
                result.artifact_path,
                result.metrics,
                metadata={
                    "dataset": dataset or IRIS_DATASET,
                    "dataset_sha256": source["sha256"] if source else None,
                    "dataset_hash": result.dataset_hash,
                    "lineage_hash": result.lineage_hash,
                    "model_card": result.model_card_path,
                    "training_run_id": run.id,
                },
            )
            run.status = "completed"
            run.metrics = result.metrics
            run.artifact_path = result.artifact_path
            run.duration_seconds = result.duration_seconds
            run.dataset_hash = result.dataset_hash
            run.lineage_hash = result.lineage_hash
            run.model_card_path = result.model_card_path
            experiment = self.tracker.log_run(
                f"{model_name}-{run.id}",
                parameters,
                {k: float(v) for k, v in result.metrics.items() if isinstance(v, int | float)},
                tags={"model": model_name, "version": str(registration["version"])},
            )
            db.add(
                ExperimentRun(
                    name=experiment["name"],
                    backend=self.tracker.backend,
                    params=experiment["params"],
                    metrics=experiment["metrics"],
                    tags=experiment["tags"],
                    external_run_id=experiment.get("run_id"),
                )
            )
            TRAINING_COUNT.labels(model_name, "completed").inc()
            prediction_cache.invalidate(model_name)
        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)
            run.duration_seconds = round(perf_counter() - start, 4)
            TRAINING_COUNT.labels(model_name, "failed").inc()
            logger.exception("training_failed", model=model_name, run_id=run.id)
        db.commit()
        db.refresh(run)
        record_audit(db, "training", model=model_name, run_id=run.id, status=run.status)
        return run

    # ----------------------------------------------------------------- inference

    def predict_batch(
        self,
        db: Session,
        items: list[tuple[str, list[float]]],
        *,
        request_id: str | None = None,
        records: dict[str, dict[str, Any]] | None = None,
        canary_id: int | None = None,
        canary_arm: str | None = None,
    ) -> list[dict]:
        """Inferência em lote: uma carga de artefato e uma predição vetorizada por modelo.

        `records` fixa a versão servida de um modelo (é como o canário escolhe o braço);
        sem ele vale `serving_record`.
        """
        canary_id, canary_arm = validate_canary_attribution(canary_id, canary_arm)
        positions: dict[str, list[int]] = {}
        for position, (model_name, _) in enumerate(items):
            positions.setdefault(model_name, []).append(position)

        logs: list[PredictionLog | None] = [None] * len(items)
        results: list[dict | None] = [None] * len(items)
        for model_name, indexes in positions.items():
            record = (records or {}).get(model_name) or self.serving_record(model_name)
            artifact = self.load(record)
            for i in indexes:
                self._check_width(artifact, record, items[i][1])
            labels = [str(label) for label in artifact["label_encoder"].classes_]
            cached_data: dict[int, tuple[str, float, dict[str, float], float]] = {}
            uncached_indexes: list[int] = []

            for i in indexes:
                cached = prediction_cache.get(model_name, record["version"], items[i][1])
                if cached is not None:
                    cached_data[i] = (cached[0], cached[1], cached[2], 0.05)
                else:
                    uncached_indexes.append(i)

            if uncached_indexes:
                started = perf_counter()
                classes, probabilities = predict_artifact(artifact, [items[i][1] for i in uncached_indexes])
                measured_latency = round((perf_counter() - started) * 1000 / len(uncached_indexes), 4)
                for offset, position in enumerate(uncached_indexes):
                    probability_map = {
                        label: round(float(p), 8) for label, p in zip(labels, probabilities[offset], strict=True)
                    }
                    confidence = max(probability_map.values())
                    predicted_class = classes[offset]
                    pred_entry = (predicted_class, confidence, probability_map)
                    prediction_cache.set(model_name, record["version"], items[position][1], pred_entry)
                    cached_data[position] = (predicted_class, confidence, probability_map, measured_latency)

            for position in indexes:
                predicted_class, confidence, probability_map, latency_ms = cached_data[position]
                logs[position] = PredictionLog(
                    model_name=model_name,
                    model_version=record["version"],
                    canary_id=canary_id,
                    canary_arm=canary_arm,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    features=items[position][1],
                    latency_ms=latency_ms,
                    request_id=request_id,
                )
                results[position] = {
                    "model_name": model_name,
                    "model_version": record["version"],
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": probability_map,
                    "request_id": request_id,
                    "latency_ms": latency_ms,
                }
                PREDICTION_COUNT.labels(model_name, predicted_class).inc()
                PREDICTION_LATENCY.labels(model_name).observe(latency_ms / 1000)
        db.add_all(logs)
        db.commit()
        for log, result in zip(logs, results, strict=True):
            result["prediction_id"] = log.id
        return results

    def predict(
        self, db: Session, model_name: str, features: list[float], *, request_id: str | None = None
    ) -> dict:
        return self.predict_batch(db, [(model_name, features)], request_id=request_id)[0]

    def score_dataframe(self, model_name: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Pontua um DataFrame fora da API (CLI `batch-score`), sem gravar logs de predição."""
        artifact = self.load(self.serving_record(model_name))
        classes, probabilities = predict_artifact(
            artifact, frame[artifact["features"]].astype(float).values.tolist()
        )
        return frame.assign(prediction=classes, confidence=probabilities.max(axis=1))

    def record_feedback(self, db: Session, prediction_id: int, actual_class: str) -> PredictionLog:
        """Registra o rótulo real de uma predição; alimenta a acurácia observada."""
        log = db.get(PredictionLog, prediction_id)
        if log is None:
            raise KeyError(f"Prediction {prediction_id} not found")
        log.actual_class = actual_class
        db.commit()
        return log

    def explain(self, model_name: str, features: list[float]) -> dict:
        """Explica a predição desta instância (Shapley exato) e traz a importância global do treino."""
        record = self.serving_record(model_name)
        artifact = self.load(record)
        self._check_width(artifact, record, features)
        background = artifact.get("background")
        if background is None:  # artefatos anteriores à 8.0 não guardam a amostra de fundo
            background = load_iris_dataframe()[FEATURE_COLUMNS].sample(
                100, random_state=self.settings.random_state
            )
        _, probabilities = predict_artifact(artifact, [features])
        class_index = int(probabilities[0].argmax())
        explanation = shapley_explanation(artifact["model"], background, features, class_index)
        explanation["predicted_class"] = str(artifact["label_encoder"].classes_[class_index])
        explanation["global_importance"] = (record.get("metrics") or {}).get("feature_importance", [])
        return {"model_name": model_name, "model_version": record["version"], "explanation": explanation}

    # ---------------------------------------------------------------------- data

    def seed_database(self, db: Session, *, replace: bool = False) -> int:
        if replace:
            db.query(IrisRecord).delete()
            db.commit()
        if db.scalar(select(IrisRecord.id).limit(1)) is not None:
            return 0
        frame = load_iris_dataframe()
        db.add_all([IrisRecord(**row, source="sklearn") for row in frame.to_dict(orient="records")])
        db.commit()
        return len(frame)

    @staticmethod
    def quality_report() -> dict:
        result = validate_training_frame(load_iris_dataframe())
        return {
            "passed": result.passed,
            "score": result.score,
            "checks": result.checks,
            "profile": result.profile,
        }

    # ---------------------------------------------------------------- monitoring

    def _recent_predictions(self, db: Session, model_name: str) -> list[PredictionLog]:
        query = (
            select(PredictionLog)
            .where(PredictionLog.model_name == model_name)
            .order_by(PredictionLog.id.desc())
            .limit(MONITORING_WINDOW)
        )
        return list(db.scalars(query).all())

    def reference_frame(self, model_name: str) -> tuple[pd.DataFrame, list[str], str]:
        """Dataset (na versão exata) em que o modelo servido foi treinado, com features e alvo."""
        record = self.serving_record(model_name)
        artifact = self.load(record)
        dataset = (artifact.get("metadata") or {}).get("dataset")
        frame, _, _ = self.dataset_frame(dataset, (record.get("metadata") or {}).get("dataset_sha256"))
        return frame, list(artifact["features"]), artifact["target"]

    def _drift(self, model_name: str, rows: list[PredictionLog]) -> dict[str, Any]:
        if len(rows) < MIN_DRIFT_SAMPLES:
            return {"status": "insufficient_data", "samples": len(rows), "required": MIN_DRIFT_SAMPLES}
        try:
            reference, features, _ = self.reference_frame(model_name)
        except ValueError as exc:  # o dataset enviado foi removido do catálogo
            return {"status": "reference_unavailable", "samples": len(rows), "detail": str(exc)}
        current = pd.DataFrame(
            [row.features for row in rows if len(row.features) == len(features)], columns=features
        )
        report = drift_report(
            reference,
            current,
            features,
            warning=self.settings.drift_psi_warning,
            critical=self.settings.drift_psi_threshold,
        )
        report["samples"] = len(rows)
        return report

    def drift_report(self, db: Session, model_name: str) -> dict:
        """PSI das features das predições recentes contra o dataset de referência."""
        report = self._drift(model_name, self._recent_predictions(db, model_name))
        return {"model_name": model_name, "status": report["status"], "report": report}

    def monitoring_summary(self, db: Session, model_name: str) -> dict[str, Any]:
        """Resumo medido das últimas predições: confiança, latência, acurácia rotulada, drift, SLOs e saúde."""
        rows = self._recent_predictions(db, model_name)
        if not rows:
            return {"model_name": model_name, "status": "no_data", "predictions": 0}
        latencies = [row.latency_ms for row in rows if row.latency_ms is not None]
        labeled = [row for row in rows if row.actual_class]
        observed: dict[str, float] = {"mean_confidence": float(np.mean([row.confidence for row in rows]))}
        if latencies:
            observed["p95_latency_ms"] = float(np.percentile(latencies, 95))
        if labeled:
            observed["accuracy_when_labeled"] = sum(
                r.predicted_class == r.actual_class for r in labeled
            ) / len(labeled)
        slo = SLOEvaluator(
            [
                SLO("p95_latency_ms", self.settings.slo_p95_latency_ms, "lte"),
                SLO("mean_confidence", self.settings.slo_min_mean_confidence, "gte"),
                SLO("accuracy_when_labeled", self.settings.gate_min_accuracy, "gte"),
            ]
        ).evaluate(observed)
        drift = self._drift(model_name, rows)
        health = score_health(
            quality_passed=self._reference_quality(model_name),
            drift_status=drift["status"],
            failed_slos=[check["name"] for check in slo["checks"] if check.get("passed") is False],
        )
        return {
            "model_name": model_name,
            "status": health["status"],
            "predictions": len(rows),
            "window": MONITORING_WINDOW,
            "labeled": len(labeled),
            "observed": {key: round(value, 6) for key, value in observed.items()},
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 6) if latencies else None,
            "class_distribution": dict(Counter(row.predicted_class for row in rows)),
            "drift": drift,
            "slo": slo,
            "health": health,
        }

    def _reference_quality(self, model_name: str) -> bool:
        try:
            frame, features, target = self.reference_frame(model_name)
        except (ValueError, ModelNotFoundError):
            return False
        iris = features == FEATURE_COLUMNS
        return validate_training_frame(frame, None if iris else features, None if iris else target).passed

    def snapshot_monitoring(self, db: Session, model_name: str) -> MonitoringReport:
        """Persiste o resumo atual em `monitoring_reports` e publica os gauges do Prometheus."""
        summary = self.monitoring_summary(db, model_name)
        report = MonitoringReport(
            model_name=model_name, report_type="monitoring", status=summary["status"], payload=summary
        )
        db.add(report)
        db.commit()
        if summary["predictions"]:
            DRIFT_COUNT.labels(summary["drift"]["status"]).inc()
            HEALTH_SCORE.labels(model_name).set(summary["health"]["score"])
            DRIFT_PSI.labels(model_name).set(summary["drift"].get("max_psi", 0.0))
        return report


@lru_cache
def get_service() -> MLService:
    """Instância única do serviço, compartilhada por todos os routers e pela CLI."""
    return MLService(get_settings())
