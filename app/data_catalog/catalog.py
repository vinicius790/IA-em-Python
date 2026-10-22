from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..platform.hash_utils import canonical_hash, sha256_file
from ..registry import file_lock
from .contracts import DatasetContract
from .profiling import profile_dataframe


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    """Registro do catálogo como a API o devolve: sem o caminho do arquivo no servidor."""
    return {key: value for key, value in record.items() if key != "path"}


class DatasetCatalog:
    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / "catalog.json"
        self._lock_path = self.root / "catalog.lock"

    def _read(self):
        return json.loads(self.catalog_path.read_text(encoding="utf-8")) if self.catalog_path.exists() else []

    def _write(self, rows):
        tmp = self.catalog_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.catalog_path)

    def register(
        self,
        name: str,
        path: str | Path,
        contract: DatasetContract | None = None,
        *,
        target: str | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        p = Path(path)
        frame = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
        validation = contract.validate(frame) if contract else {"passed": True, "checks": []}
        if not validation["passed"]:
            raise ValueError("Dataset contract failed")
        checksum = sha256_file(p)
        record = {
            "id": canonical_hash({"name": name, "checksum": checksum})[:16],
            "name": name,
            "path": str(p),
            "sha256": checksum,
            "rows": len(frame),
            "profile": profile_dataframe(frame),
            "contract": contract.as_dict() if contract else None,
            "target": target,
            "features": features,
            "created_at": datetime.now(UTC).isoformat(),
        }
        # Ler-alterar-gravar sob trava de arquivo: uploads chegam em paralelo pelo threadpool e
        # podem vir de outros processos (vários workers do uvicorn, a CLI) no mesmo `DATA_DIR`.
        with file_lock(self._lock_path):
            records = [r for r in self._read() if r["id"] != record["id"]]
            records.append(record)
            self._write(records)
        return record

    def list(self, name: str | None = None):
        return [r for r in self._read() if name is None or r["name"] == name]

    def get(self, name: str) -> dict[str, Any] | None:
        """Versão mais recente registrada com este nome."""
        return next((r for r in reversed(self._read()) if r["name"] == name), None)
