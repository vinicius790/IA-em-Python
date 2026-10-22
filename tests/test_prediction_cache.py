"""Testes unitários e de integração para o Cache de Predições de Baixa Latência."""

import concurrent.futures
import time

import pytest

from app.platform.cache import PredictionCache, cached_prediction, make_cache_key


def test_prediction_cache_basic_get_set_miss():
    cache = PredictionCache(max_size=5, ttl_seconds=60.0)

    # Miss inicial
    assert cache.get("iris_model", 1, [5.1, 3.5, 1.4, 0.2]) is None

    # Set e Hit
    cache.set("iris_model", 1, [5.1, 3.5, 1.4, 0.2], {"class": "setosa", "confidence": 0.98})
    result = cache.get("iris_model", 1, [5.1, 3.5, 1.4, 0.2])

    assert result == {"class": "setosa", "confidence": 0.98}
    assert cache.size == 1

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_prediction_cache_ttl_expiration():
    # Cache com TTL muito curto (0.05 segundos)
    cache = PredictionCache(max_size=10, ttl_seconds=0.05)
    features = [6.0, 3.0, 4.8, 1.8]

    cache.set("rf_model", "v1", features, "virginica")
    assert cache.get("rf_model", "v1", features) == "virginica"

    # Aguarda expirar
    time.sleep(0.06)
    assert cache.get("rf_model", "v1", features) is None
    assert cache.size == 0


def test_prediction_cache_lru_eviction():
    cache = PredictionCache(max_size=3, ttl_seconds=60.0)

    cache.set("m", 1, [1.0], "res1")
    cache.set("m", 1, [2.0], "res2")
    cache.set("m", 1, [3.0], "res3")

    # Acessa [1.0] tornando-o mais recente que [2.0]
    assert cache.get("m", 1, [1.0]) == "res1"

    # Inserir 4º elemento deve expulsar o menos recentemente usado ([2.0])
    cache.set("m", 1, [4.0], "res4")

    assert cache.get("m", 1, [2.0]) is None  # expulso
    assert cache.get("m", 1, [1.0]) == "res1"
    assert cache.get("m", 1, [3.0]) == "res3"
    assert cache.get("m", 1, [4.0]) == "res4"
    assert cache.stats()["evictions"] == 1


def test_prediction_cache_invalidation():
    cache = PredictionCache(max_size=10)

    cache.set("model_a", 1, [1.0], "a1")
    cache.set("model_a", 2, [1.0], "a2")
    cache.set("model_b", 1, [1.0], "b1")

    # Invalida apenas model_a versão 1
    deleted = cache.invalidate("model_a", 1)
    assert deleted == 1
    assert cache.get("model_a", 1, [1.0]) is None
    assert cache.get("model_a", 2, [1.0]) == "a2"
    assert cache.get("model_b", 1, [1.0]) == "b1"

    # Invalida todo o model_a
    deleted = cache.invalidate("model_a")
    assert deleted == 1
    assert cache.get("model_a", 2, [1.0]) is None
    assert cache.get("model_b", 1, [1.0]) == "b1"

    # Invalida tudo
    cache.clear()
    assert cache.size == 0


def test_cached_prediction_decorator():
    cache = PredictionCache(max_size=10)
    evaluations = 0

    @cached_prediction(cache=cache)
    def heavy_inference(model_name: str, model_version: int | str, features: list[float]) -> str:
        nonlocal evaluations
        evaluations += 1
        return f"pred_{features[0]}"

    feat = [5.5, 2.3, 4.0, 1.3]
    # 1ª chamada: executa a função
    r1 = heavy_inference("knn", 1, feat)
    assert r1 == "pred_5.5"
    assert evaluations == 1

    # 2ª chamada idêntica: retorna do cache sem re-executar
    r2 = heavy_inference("knn", 1, feat)
    assert r2 == "pred_5.5"
    assert evaluations == 1

    # Features diferentes: executa
    r3 = heavy_inference("knn", 1, [7.0, 3.2, 4.7, 1.4])
    assert r3 == "pred_7.0"
    assert evaluations == 2


def test_prediction_cache_thread_safety():
    cache = PredictionCache(max_size=200, ttl_seconds=10.0)

    def worker(worker_id: int):
        for i in range(100):
            features = [float(worker_id), float(i % 10)]
            cache.set("multi_thread_model", 1, features, {"worker": worker_id, "idx": i})
            res = cache.get("multi_thread_model", 1, features)
            assert res is not None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, w) for w in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert cache.size <= 200
    assert cache.stats()["hits"] >= 800
