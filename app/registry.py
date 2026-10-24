from __future__ import annotations

import hmac
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .platform.hash_utils import sha256_file

STAGES = {"candidate", "staging", "production", "archived"}
LOCK_TIMEOUT_SECONDS = 15.0
STALE_LOCK_SECONDS = 60.0

_THREAD_LOCK = threading.Lock()


@contextmanager
def file_lock(lock_path: Path) -> Iterator[None]:
    """Trava exclusiva entre threads e processos da mesma máquina, por arquivo (`O_EXCL`).

    ponytail: lock por arquivo com expiração; em disco de rede ou com várias máquinas,
    mova o estado para o banco de dados.
    """
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    with _THREAD_LOCK:
        while True:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except (FileExistsError, PermissionError):
                try:
                    if time.time() - lock_path.stat().st_mtime > STALE_LOCK_SECONDS:
                        lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Could not lock {lock_path}") from None
                time.sleep(0.02)
        try:
            yield
        finally:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)


class ModelRegistry:
    """Registry em arquivo: versões, estágios, checksum SHA-256 e assinatura opcional.

    Os artefatos ficam sempre dentro de `directory`; o catálogo guarda só o nome do
    arquivo, então o registry não depende do diretório de trabalho do processo.
    """

    def __init__(self, directory: str | Path, signing_key: str = "") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.directory / "catalog.json"
        self._lock_path = self.directory / "catalog.lock"
        self._signing_key = signing_key.encode()

    def _locked(self) -> AbstractContextManager[None]:
        return file_lock(self._lock_path)

    def _read(self) -> list[dict[str, Any]]:
        if not self.catalog_path.exists():
            return []
        return json.loads(self.catalog_path.read_text(encoding="utf-8"))

    def _write(self, records: list[dict[str, Any]]) -> None:
        temp = self.catalog_path.with_suffix(".tmp")
        temp.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        temp.replace(self.catalog_path)

    def _sign(self, record: dict[str, Any]) -> str:
        message = f"{record['name']}:{record['version']}:{record['checksum_sha256']}".encode()
        return hmac.new(self._signing_key, message, "sha256").hexdigest()

    def artifact_file(self, record: dict[str, Any]) -> Path:
        """Caminho do artefato; só o nome do arquivo é usado, sempre dentro do registry."""
        return self.directory / Path(record["artifact_path"]).name

    def register(
        self,
        model_name: str,
        artifact_path: str | Path,
        metrics: dict,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = Path(artifact_path)
        if source.resolve().parent != self.directory.resolve():
            raise ValueError(f"Artifact must be inside the registry directory: {source}")
        with self._locked():
            records = self._read()
            version = max((r["version"] for r in records if r["name"] == model_name), default=0) + 1
            record: dict[str, Any] = {
                "name": model_name,
                "version": version,
                "alias": f"{model_name}:v{version}",
                "stage": "candidate",
                "artifact_path": source.name,
                "checksum_sha256": sha256_file(source),
                "metrics": metrics,
                "metadata": metadata or {},
                "created_at": datetime.now(UTC).isoformat(),
            }
            if self._signing_key:
                record["signature"] = self._sign(record)
            records.append(record)
            self._write(records)
        return record

    def list(self) -> list[dict[str, Any]]:
        return list(reversed(self._read()))

    def get(self, model_name: str, version: int) -> dict[str, Any] | None:
        return next((r for r in self._read() if r["name"] == model_name and r["version"] == version), None)

    def latest(self, model_name: str) -> dict[str, Any] | None:
        return next(
            (r for r in self.list() if r["name"] == model_name and r.get("stage") != "archived"), None
        )

    def production(self, model_name: str) -> dict[str, Any] | None:
        return next(
            (r for r in self.list() if r["name"] == model_name and r.get("stage") == "production"), None
        )

    def promote(self, model_name: str, version: int, stage: str, *, notes: str = "") -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Use one of: {', '.join(sorted(STAGES))}")
        with self._locked():
            records = self._read()
            target = next((r for r in records if r["name"] == model_name and r["version"] == version), None)
            if target is None:
                raise KeyError(f"Model {model_name} version {version} not found")
            if stage == "production":
                for record in records:
                    if record["name"] == model_name and record.get("stage") == "production":
                        record["stage"] = "archived"
            target["stage"] = stage
            target["notes"] = notes
            target["promoted_at"] = datetime.now(UTC).isoformat()
            self._write(records)
        return target

    def _sign_onnx(self, record: dict[str, Any]) -> str:
        message = f"{record['name']}:{record['version']}:onnx:{record['onnx_checksum_sha256']}".encode()
        return hmac.new(self._signing_key, message, "sha256").hexdigest()

    def onnx_file(self, record: dict[str, Any]) -> Path | None:
        """Caminho do ``model.onnx`` opcional da versão, sempre dentro do registry."""
        name = record.get("onnx_artifact_path")
        return self.directory / Path(name).name if name else None

    def attach_onnx(self, model_name: str, version: int, onnx_path: str | Path) -> dict[str, Any]:
        """Anexa o ``model.onnx`` à versão, com SHA-256 (e assinatura) próprios."""
        source = Path(onnx_path)
        if source.resolve().parent != self.directory.resolve():
            raise ValueError(f"Artifact must be inside the registry directory: {source}")
        with self._locked():
            records = self._read()
            target = next((r for r in records if r["name"] == model_name and r["version"] == version), None)
            if target is None:
                raise KeyError(f"Model {model_name} version {version} not found")
            target["onnx_artifact_path"] = source.name
            target["onnx_checksum_sha256"] = sha256_file(source)
            if self._signing_key:
                target["onnx_signature"] = self._sign_onnx(target)
            self._write(records)
        return target

    def set_serving_backend(self, model_name: str, version: int, backend: str) -> dict[str, Any]:
        """Define o backend de inferência da versão: ``sklearn`` (padrão) ou ``onnx``."""
        if backend not in ("sklearn", "onnx"):
            raise ValueError(f"Invalid backend '{backend}'. Use one of: onnx, sklearn")
        with self._locked():
            records = self._read()
            target = next((r for r in records if r["name"] == model_name and r["version"] == version), None)
            if target is None:
                raise KeyError(f"Model {model_name} version {version} not found")
            if backend == "onnx" and not target.get("onnx_artifact_path"):
                raise ValueError(f"{target['alias']} has no ONNX artifact; attach one first")
            target["serving_backend"] = backend
            self._write(records)
        return target

    def verify_onnx(self, record: dict[str, Any]) -> bool:
        """Mesma verificação do pickle (checksum + assinatura HMAC), para o ``model.onnx``."""
        path = self.onnx_file(record)
        if path is None or not path.is_file() or sha256_file(path) != record.get("onnx_checksum_sha256"):
            return False
        if not self._signing_key:
            return True
        return hmac.compare_digest(str(record.get("onnx_signature", "")), self._sign_onnx(record))

    def verify(self, record: dict[str, Any]) -> bool:
        """Confere o checksum e, se houver chave configurada, a assinatura HMAC.

        O checksum sozinho detecta corrupção, mas não protege contra quem consegue
        escrever no diretório (o catálogo fica ao lado do artefato). A assinatura,
        feita com `ARTIFACT_SIGNING_KEY`, fecha essa brecha.
        """
        path = self.artifact_file(record)
        if not path.is_file() or sha256_file(path) != record.get("checksum_sha256"):
            return False
        if not self._signing_key:
            return True
        return hmac.compare_digest(str(record.get("signature", "")), self._sign(record))
