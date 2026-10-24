"""Registry de modelos, backend de serving e promoção (fatia de ``MLService``, issue #9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..deployment.gates import evaluate_release_gate
from ..errors import ArtifactIntegrityError, GateFailedError, ModelNotFoundError
from ..governance.audit import record_audit
from ..ml import load_artifact, predict_artifact
from ..serving.onnx_backend import OnnxPredictor, export_onnx, predict_onnx, resolve_backend
from ..tracing import set_span_attributes, traced
from ._base import ServiceBase


class RegistryMixin(ServiceBase):
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

    def predict_rows(
        self, record: dict[str, Any], artifact: dict[str, Any], rows: list[list[float]]
    ) -> tuple[list[str], Any]:
        """Prediz pelo backend da versão (`serving_backend`); fallback para sklearn é logado."""
        decision = resolve_backend(self.registry, record)
        if decision.backend == "onnx" and decision.onnx_path is not None:
            key = str(decision.onnx_path)
            stamp = (record.get("onnx_checksum_sha256"), decision.onnx_path.stat().st_mtime_ns)
            cached = self._onnx_sessions.get(key)
            if not cached or cached[0] != stamp:
                cached = (stamp, OnnxPredictor(decision.onnx_path))
                self._onnx_sessions[key] = cached
            return predict_onnx(cached[1], artifact, rows)
        return predict_artifact(artifact, rows)

    def export_onnx(self, model_name: str, version: int, *, activate: bool = False) -> dict[str, Any]:
        """Exporta a versão para ``model.onnx`` no registry; com `activate`, serve via ONNX."""
        record = self.registry.get(model_name, version)
        if record is None:
            raise ModelNotFoundError(f"Model {model_name} version {version} not found")
        artifact = self.load(record)
        target = self.registry.directory / f"{Path(record['artifact_path']).stem}.onnx"
        export_onnx(artifact, target)
        record = self.registry.attach_onnx(model_name, version, target)
        if activate:
            record = self.registry.set_serving_backend(model_name, version, "onnx")
        return record

    def release_gate(self, record: dict[str, Any]) -> dict[str, Any]:
        return evaluate_release_gate(
            record.get("metrics") or {},
            min_accuracy=self.settings.gate_min_accuracy,
            min_f1=self.settings.gate_min_f1,
            max_latency_ms=self.settings.gate_max_latency_ms,
        )

    @traced("iris.promote")
    def promote(
        self, db: Session, model_name: str, version: int, stage: str, *, notes: str = "", force: bool = False
    ) -> dict[str, Any]:
        """Muda o estágio; ir para `production` exige passar no release gate (ou `force`)."""
        set_span_attributes(model_name=model_name, model_version=version, stage=stage, forced=force)
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
