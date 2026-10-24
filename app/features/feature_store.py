"""Feature Store Corporativa com Online Store em memória e Offline Store histórica.

Suporta:
- Definição tipada de Feature Views com metadados e TTL.
- Online Store de alta performance para inferência em tempo real.
- Offline Store para geração de datasets históricos de treino sem data leakage.
- Linhagem de features e versionamento semântico.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadados de uma feature individual."""

    name: str
    dtype: str
    description: str = ""
    unit: str | None = None
    nullable: bool = False


@dataclass
class FeatureView:
    """Visão de features corporativa com metadados, linhagem e versionamento."""

    name: str
    version: str = "v1"
    entity_keys: list[str] = field(default_factory=lambda: ["id"])
    features: list[FeatureMetadata] = field(default_factory=list)
    description: str = ""
    author: str = "ai-team"
    ttl_seconds: int | None = 86400  # 24 horas por padrão
    source_dataset: str = "iris"
    transformation_logic: str = "identity"
    tags: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def view_id(self) -> str:
        return f"{self.name}:{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "view_id": self.view_id,
            "entity_keys": self.entity_keys,
            "features": [
                {
                    "name": f.name,
                    "dtype": f.dtype,
                    "description": f.description,
                    "unit": f.unit,
                    "nullable": f.nullable,
                }
                for f in self.features
            ],
            "description": self.description,
            "author": self.author,
            "ttl_seconds": self.ttl_seconds,
            "source_dataset": self.source_dataset,
            "transformation_logic": self.transformation_logic,
            "tags": self.tags,
            "created_at": self.created_at,
        }


class OnlineStore:
    """Online Store thread-safe em memória para inferência com latência em milissegundos."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # {view_id: {entity_key_str: (values_dict, timestamp_epoch)}}
        self._store: dict[str, dict[str, tuple[dict[str, Any], float]]] = {}
        self._stats = {"reads": 0, "hits": 0, "misses": 0, "writes": 0}

    @staticmethod
    def _entity_key(entity: dict[str, Any] | Any, key_names: list[str]) -> str:
        if isinstance(entity, dict):
            return ":".join(str(entity.get(k, "")) for k in key_names)
        return str(entity)

    def write(
        self,
        view_id: str,
        records: list[dict[str, Any]],
        key_names: list[str],
        timestamp: float | None = None,
    ) -> int:
        """Grava registros na online store com timestamp de atualização."""
        now = timestamp if timestamp is not None else time.time()
        with self._lock:
            if view_id not in self._store:
                self._store[view_id] = {}
            target = self._store[view_id]
            for record in records:
                key = self._entity_key(record, key_names)
                target[key] = (record.copy(), now)
            self._stats["writes"] += len(records)
            return len(records)

    def read(
        self,
        view_id: str,
        entity_keys: list[dict[str, Any] | Any],
        key_names: list[str],
        feature_names: list[str] | None = None,
        ttl_seconds: int | None = None,
    ) -> list[dict[str, Any] | None]:
        """Recupera features por chave de entidade em milissegundos."""
        now = time.time()
        results: list[dict[str, Any] | None] = []

        with self._lock:
            view_data = self._store.get(view_id, {})
            self._stats["reads"] += len(entity_keys)

            for ek in entity_keys:
                key_str = self._entity_key(ek, key_names)
                entry = view_data.get(key_str)

                if entry is None:
                    self._stats["misses"] += 1
                    results.append(None)
                    continue

                values, write_time = entry
                if ttl_seconds is not None and (now - write_time) > ttl_seconds:
                    self._stats["misses"] += 1
                    results.append(None)
                    continue

                self._stats["hits"] += 1
                if feature_names is not None:
                    filtered = {k: values.get(k) for k in feature_names if k in values}
                    results.append(filtered)
                else:
                    results.append(values.copy())

        return results

    def clear(self, view_id: str | None = None) -> None:
        with self._lock:
            if view_id:
                self._store.pop(view_id, None)
            else:
                self._store.clear()

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            total_records = sum(len(v) for v in self._store.values())
            return {**self._stats, "stored_records": total_records, "view_count": len(self._store)}


class OfflineStore:
    """Offline Store para geração de datasets históricos de treino sem data leakage."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # {view_id: pd.DataFrame}
        self._data: dict[str, pd.DataFrame] = {}

    def write(self, view_id: str, frame: pd.DataFrame, timestamp_col: str = "timestamp") -> int:
        """Ingere dados históricos com suporte a append."""
        with self._lock:
            df = frame.copy()
            if timestamp_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

            if view_id in self._data:
                self._data[view_id] = pd.concat([self._data[view_id], df], ignore_index=True)
            else:
                self._data[view_id] = df
            return len(df)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        view_id: str,
        feature_names: list[str] | None = None,
        entity_keys: list[str] | None = None,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Point-in-time (as-of) join para evitar data leakage em datasets de treino.

        Para cada linha de `entity_df`, busca os valores de features mais recentes
        com timestamp <= entity_df[timestamp_col].
        """
        with self._lock:
            if view_id not in self._data:
                raise ValueError(f"Feature view '{view_id}' não encontrada na Offline Store")

            feature_df = self._data[view_id].copy()

        keys = entity_keys or ["id"]
        features_to_select = feature_names or [
            c for c in feature_df.columns if c not in keys and c != timestamp_col
        ]

        # Se não houver coluna de timestamp, faz merge simples nas entity_keys
        if timestamp_col not in entity_df.columns or timestamp_col not in feature_df.columns:
            cols = list(dict.fromkeys(keys + features_to_select))
            return pd.merge(entity_df, feature_df[cols], on=keys, how="left")

        # Conversão para datetime para merge_asof
        ent = entity_df.copy()
        feat = feature_df.copy()

        ent[timestamp_col] = pd.to_datetime(ent[timestamp_col], utc=True)
        feat[timestamp_col] = pd.to_datetime(feat[timestamp_col], utc=True)

        # Ordenar ambas pelo timestamp requerido por pd.merge_asof
        ent = ent.sort_values(by=timestamp_col)
        feat = feat.sort_values(by=timestamp_col)

        cols_feat = list(dict.fromkeys(keys + [timestamp_col] + features_to_select))
        joined = pd.merge_asof(
            ent,
            feat[cols_feat],
            on=timestamp_col,
            by=keys[0] if len(keys) == 1 else None,
            direction="backward",
        )
        return joined


class FeatureStore:
    """Fachada unificada da Feature Store Corporativa.

    Combina catálogo de metadados, linhagem, online store e offline store.
    """

    def __init__(self) -> None:
        self._views: dict[str, FeatureView] = {}
        self._lineage: dict[str, dict[str, Any]] = {}
        self.online_store = OnlineStore()
        self.offline_store = OfflineStore()
        self._lock = threading.RLock()

    def register_view(self, view: FeatureView) -> FeatureView:
        """Registra ou atualiza uma Feature View no catálogo com rastreamento de versão."""
        with self._lock:
            self._views[view.view_id] = view
            # Grava linhagem
            self._lineage[view.view_id] = {
                "view_id": view.view_id,
                "name": view.name,
                "version": view.version,
                "source_dataset": view.source_dataset,
                "transformation_logic": view.transformation_logic,
                "features": view.feature_names,
                "created_at": view.created_at,
                "author": view.author,
            }
            return view

    def get_view(self, name: str, version: str | None = None) -> FeatureView | None:
        """Obtém uma view por nome e versão (se omitida, retorna a mais recente cadastrada)."""
        with self._lock:
            if version:
                return self._views.get(f"{name}:{version}")
            # Busca última versão por prefixo
            matching = [v for k, v in self._views.items() if v.name == name]
            return matching[-1] if matching else None

    def list_views(self) -> list[dict[str, Any]]:
        with self._lock:
            return [v.to_dict() for v in self._views.values()]

    def get_lineage(self, view_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._lineage.get(view_id)

    # Métodos de conveniência Online
    def ingest_online(self, view_name: str, records: list[dict[str, Any]], version: str | None = None) -> int:
        view = self.get_view(view_name, version)
        if not view:
            raise ValueError(f"Feature view '{view_name}' não encontrada.")
        return self.online_store.write(view.view_id, records, view.entity_keys)

    def lookup_online(
        self,
        view_name: str,
        entity_keys: list[dict[str, Any] | Any],
        features: list[str] | None = None,
        version: str | None = None,
    ) -> list[dict[str, Any] | None]:
        """Recupera features da online store com checagem de TTL."""
        view = self.get_view(view_name, version)
        if not view:
            raise ValueError(f"Feature view '{view_name}' não encontrada.")
        return self.online_store.read(
            view.view_id,
            entity_keys,
            view.entity_keys,
            feature_names=features,
            ttl_seconds=view.ttl_seconds,
        )

    # Métodos de conveniência Offline
    def ingest_offline(
        self,
        view_name: str,
        frame: pd.DataFrame,
        version: str | None = None,
        timestamp_col: str = "timestamp",
    ) -> int:
        view = self.get_view(view_name, version)
        if not view:
            raise ValueError(f"Feature view '{view_name}' não encontrada.")
        return self.offline_store.write(view.view_id, frame, timestamp_col=timestamp_col)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        view_name: str,
        features: list[str] | None = None,
        version: str | None = None,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        view = self.get_view(view_name, version)
        if not view:
            raise ValueError(f"Feature view '{view_name}' não encontrada.")
        return self.offline_store.get_historical_features(
            entity_df=entity_df,
            view_id=view.view_id,
            feature_names=features or view.feature_names,
            entity_keys=view.entity_keys,
            timestamp_col=timestamp_col,
        )


# Singleton padrão corporativo
_default_store: FeatureStore | None = None
_store_lock = threading.Lock()


def get_feature_store() -> FeatureStore:
    """Retorna o singleton da Feature Store inicializado com as views padrões do projeto."""
    global _default_store
    if _default_store is None:
        with _store_lock:
            if _default_store is None:
                store = FeatureStore()
                # Registrar Feature Views padrão do ecossistema Iris
                iris_measurements = FeatureView(
                    name="iris_measurements",
                    version="v1",
                    entity_keys=["iris_id"],
                    features=[
                        FeatureMetadata("sepal_length", "float64", "Comprimento da sépala em cm"),
                        FeatureMetadata("sepal_width", "float64", "Largura da sépala em cm"),
                        FeatureMetadata("petal_length", "float64", "Comprimento da pétala em cm"),
                        FeatureMetadata("petal_width", "float64", "Largura da pétala em cm"),
                    ],
                    description="Medidas morfológicas brutas da flor Iris",
                    author="ml-team",
                    ttl_seconds=86400,
                    source_dataset="iris_records",
                    transformation_logic="raw",
                    tags={"domain": "botany", "environment": "production"},
                )
                store.register_view(iris_measurements)

                iris_engineered = FeatureView(
                    name="iris_engineered",
                    version="v1",
                    entity_keys=["iris_id"],
                    features=[
                        FeatureMetadata("sepal_area", "float64", "Área aproximada da sépala"),
                        FeatureMetadata("petal_area", "float64", "Área aproximada da pétala"),
                        FeatureMetadata(
                            "sepal_aspect_ratio", "float64", "Razão comprimento/largura da sépala"
                        ),
                        FeatureMetadata(
                            "petal_aspect_ratio", "float64", "Razão comprimento/largura da pétala"
                        ),
                    ],
                    description="Atributos matematicamente derivados e interações",
                    author="mlops-architect",
                    ttl_seconds=86400,
                    source_dataset="iris_records",
                    transformation_logic="app.features.engineering.enrich_iris",
                    tags={"domain": "features", "tier": "gold"},
                )
                store.register_view(iris_engineered)
                _default_store = store
    return _default_store
