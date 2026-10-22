"""Testes automatizados da Feature Store Corporativa (Online, Offline, Linhagem e Versionamento)."""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.features.feature_store import (
    FeatureMetadata,
    FeatureStore,
    FeatureView,
    OfflineStore,
    OnlineStore,
    get_feature_store,
)


def test_feature_view_definition_and_metadata():
    view = FeatureView(
        name="customer_credit_features",
        version="v1",
        entity_keys=["customer_id"],
        features=[
            FeatureMetadata("income", "float64", "Renda mensal estimada", unit="BRL"),
            FeatureMetadata("credit_score", "int64", "Pontuação de crédito", unit="points"),
        ],
        description="Features de risco financeiro do cliente",
        author="mlops-engineer",
        ttl_seconds=3600,
        source_dataset="credit_bureau_stream",
        transformation_logic="sql_aggregation",
        tags={"compliance": "lgpd", "criticality": "high"},
    )

    data = view.to_dict()
    assert data["name"] == "customer_credit_features"
    assert data["version"] == "v1"
    assert data["view_id"] == "customer_credit_features:v1"
    assert data["entity_keys"] == ["customer_id"]
    assert len(data["features"]) == 2
    assert data["features"][0]["name"] == "income"
    assert data["ttl_seconds"] == 3600
    assert view.feature_names == ["income", "credit_score"]


def test_online_store_ingestion_and_submillisecond_lookup():
    online = OnlineStore()
    view_id = "user_features:v1"
    keys = ["user_id"]

    records = [
        {"user_id": 101, "age": 28, "income": 5400.0},
        {"user_id": 102, "age": 45, "income": 12000.0},
    ]

    written = online.write(view_id, records, key_names=keys)
    assert written == 2

    # Recuperação unitária e em lote
    start = time.perf_counter()
    results = online.read(view_id, [101, 102, 999], key_names=keys)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(results) == 3
    assert results[0]["age"] == 28
    assert results[1]["income"] == 12000.0
    assert results[2] is None  # Miss
    assert elapsed_ms < 10.0  # Recuperação ultrarrápida em memória

    # Estatísticas de leitura
    stats = online.stats
    assert stats["reads"] == 3
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["stored_records"] == 2


def test_online_store_feature_filtering():
    online = OnlineStore()
    view_id = "v:1"
    keys = ["id"]
    online.write(view_id, [{"id": 1, "f1": 10, "f2": 20, "f3": 30}], key_names=keys)

    filtered = online.read(view_id, [1], key_names=keys, feature_names=["f1", "f3"])
    assert filtered[0] == {"f1": 10, "f3": 30}


def test_online_store_ttl_expiration():
    online = OnlineStore()
    view_id = "sensor:v1"
    keys = ["sensor_id"]

    past_timestamp = time.time() - 100.0  # Gravado há 100 segundos
    online.write(
        view_id,
        [{"sensor_id": "s1", "temp": 24.5}],
        key_names=keys,
        timestamp=past_timestamp,
    )

    # Com TTL de 50s, deve expirar e retornar None
    expired = online.read(view_id, ["s1"], key_names=keys, ttl_seconds=50)
    assert expired[0] is None

    # Com TTL de 200s, ainda deve ser válido
    valid = online.read(view_id, ["s1"], key_names=keys, ttl_seconds=200)
    assert valid[0] is not None
    assert valid[0]["temp"] == 24.5


def test_offline_store_point_in_time_correctness_without_leakage():
    offline = OfflineStore()
    view_id = "account_balance:v1"

    # Histórico de saldo que muda com o tempo
    feature_history = pd.DataFrame(
        [
            {"acc_id": 1, "balance": 100.0, "timestamp": "2026-01-01 10:00:00"},
            {"acc_id": 1, "balance": 150.0, "timestamp": "2026-01-01 12:00:00"},
            {"acc_id": 1, "balance": 300.0, "timestamp": "2026-01-01 16:00:00"},  # Evento futuro
        ]
    )
    offline.write(view_id, feature_history, timestamp_col="timestamp")

    # Dataset de eventos de transação observados pelos cientistas de dados
    observations = pd.DataFrame(
        [
            {"acc_id": 1, "tx_id": "t1", "timestamp": "2026-01-01 11:00:00"},  # Deve pegar saldo=100
            {"acc_id": 1, "tx_id": "t2", "timestamp": "2026-01-01 14:00:00"},  # Deve pegar saldo=150
        ]
    )

    historical_features = offline.get_historical_features(
        entity_df=observations,
        view_id=view_id,
        feature_names=["balance"],
        entity_keys=["acc_id"],
        timestamp_col="timestamp",
    )

    assert len(historical_features) == 2
    row1 = historical_features[historical_features["tx_id"] == "t1"].iloc[0]
    row2 = historical_features[historical_features["tx_id"] == "t2"].iloc[0]

    # Prevenção rigorosa de vazamento temporal (data leakage)
    assert row1["balance"] == 100.0
    assert row2["balance"] == 150.0


def test_feature_store_facade_and_lineage():
    store = FeatureStore()
    view = FeatureView(
        name="iris_morphology",
        version="v2",
        entity_keys=["iris_id"],
        features=[FeatureMetadata("petal_length", "float64")],
        source_dataset="iris_records",
        transformation_logic="clean_and_scale",
        author="mlops",
    )
    store.register_view(view)

    retrieved = store.get_view("iris_morphology", "v2")
    assert retrieved is not None
    assert retrieved.view_id == "iris_morphology:v2"

    lineage = store.get_lineage("iris_morphology:v2")
    assert lineage is not None
    assert lineage["source_dataset"] == "iris_records"
    assert lineage["transformation_logic"] == "clean_and_scale"
    assert lineage["author"] == "mlops"

    # Ingestão e consulta de alto nível
    store.ingest_online("iris_morphology", [{"iris_id": 1, "petal_length": 1.4}], version="v2")
    lookup = store.lookup_online("iris_morphology", [1], version="v2")
    assert lookup[0]["petal_length"] == 1.4


def test_default_feature_store_singleton_has_iris_views():
    store = get_feature_store()
    views = store.list_views()
    names = {v["name"] for v in views}
    assert "iris_measurements" in names
    assert "iris_engineered" in names
