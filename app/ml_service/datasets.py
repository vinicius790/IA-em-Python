"""Catálogo de datasets enviados (fatia de ``MLService``, issue #9)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from ..data import load_iris_dataframe
from ..data_catalog.catalog import DatasetCatalog
from ..data_catalog.csv_profile import is_null, profile_csv, read_rows, training_problems, valid_dataset_name
from ..governance.audit import record_audit
from ._base import IRIS_DATASET, ServiceBase


class DatasetsMixin(ServiceBase):
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
