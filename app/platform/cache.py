"""Cache de predições de baixa latência em memória para modelos de ML.

Evita recomputação de inferências para as mesmas features em um mesmo modelo/versão.
Implementado com OrderedDict para expulsão LRU O(1), suporte a TTL e thread-safe com RLock.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

from app.platform.hash_utils import canonical_hash

DEFAULT_MAX_SIZE = 10_000
DEFAULT_TTL_SECONDS = 300.0  # 5 minutos


@dataclass(slots=True)
class CacheEntry:
    """Entrada individual do cache."""

    value: Any
    expires_at: float | None  # time.monotonic()
    created_at: float
    hits: int = 0


def make_cache_key(
    model_name: str,
    model_version: int | str,
    features: Sequence[float | int] | Any,
    precision: int = 8,
) -> tuple[str, str, Any]:
    """Gera chave de cache determinística de alta performance para inferência."""
    try:
        # Se for sequência de números (o caso padrão de features do Iris e modelos tabulares)
        feat_tuple = tuple(round(float(x), precision) for x in features)
        return (model_name, str(model_version), feat_tuple)
    except (TypeError, ValueError):
        # Fallback para estruturas complexas ou dicionários
        return (model_name, str(model_version), canonical_hash(features))


class PredictionCache:
    """Cache LRU com TTL e métricas de desempenho para predições."""

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl_seconds: float | None = DEFAULT_TTL_SECONDS,
        *,
        precision: int = 8,
    ) -> None:
        self.max_size = max(1, max_size)
        self.ttl_seconds = ttl_seconds
        self.precision = precision

        self._cache: OrderedDict[tuple, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def get(
        self,
        model_name: str,
        model_version: int | str,
        features: Sequence[float | int] | Any,
    ) -> Any | None:
        """Recupera valor do cache.

        Retorna None em caso de miss ou expiração.
        """
        key = make_cache_key(model_name, model_version, features, self.precision)
        now = time.monotonic()

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]
            # Verifica expiração por TTL
            if entry.expires_at is not None and now >= entry.expires_at:
                del self._cache[key]
                self._misses += 1
                return None

            # Move para o fim do OrderedDict (mais recentemente usado)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            return entry.value

    def set(
        self,
        model_name: str,
        model_version: int | str,
        features: Sequence[float | int] | Any,
        value: Any,
        *,
        ttl_seconds: float | None = ...,  # type: ignore
    ) -> None:
        """Armazena um resultado de inferência no cache."""
        key = make_cache_key(model_name, model_version, features, self.precision)
        now = time.monotonic()

        effective_ttl = self.ttl_seconds if ttl_seconds is ... else ttl_seconds
        expires_at = (now + effective_ttl) if effective_ttl is not None else None

        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self.max_size:
                # Expulsa o item menos recentemente usado (primeiro item)
                self._cache.popitem(last=False)
                self._evictions += 1

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at,
                created_at=now,
            )

    def get_or_compute(
        self,
        model_name: str,
        model_version: int | str,
        features: Sequence[float | int] | Any,
        compute_fn: Callable[[], Any],
        *,
        ttl_seconds: float | None = ...,  # type: ignore
    ) -> Any:
        """Busca no cache ou computa e armazena se não existir."""
        val = self.get(model_name, model_version, features)
        if val is not None:
            return val
        computed = compute_fn()
        self.set(model_name, model_version, features, computed, ttl_seconds=ttl_seconds)
        return computed

    def invalidate(
        self,
        model_name: str | None = None,
        model_version: int | str | None = None,
    ) -> int:
        """Invalida entradas do cache filtradas por nome e/ou versão do modelo."""
        with self._lock:
            if model_name is None and model_version is None:
                count = len(self._cache)
                self._cache.clear()
                return count

            version_str = str(model_version) if model_version is not None else None
            keys_to_delete = [
                key
                for key in self._cache
                if (model_name is None or key[0] == model_name)
                and (version_str is None or key[1] == version_str)
            ]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    def clear(self) -> None:
        """Limpa todo o cache e reseta contadores de sessão."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def stats(self) -> dict[str, Any]:
        """Retorna estatísticas operacionais do cache."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = round(self._hits / total, 4) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "ttl_seconds": self.ttl_seconds,
            }

    def prune_expired(self) -> int:
        """Remove explicitamente entradas que já expiraram."""
        now = time.monotonic()
        count = 0
        with self._lock:
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if entry.expires_at is not None and now >= entry.expires_at
            ]
            for key in expired_keys:
                del self._cache[key]
                count += 1
        return count


# Instância global compartilhada na plataforma
prediction_cache = PredictionCache()


def cached_prediction(
    cache: PredictionCache | None = None,
    *,
    ttl_seconds: float | None = None,
) -> Callable:
    """Decorator para funções de predição com assinatura (model_name, model_version, features, ...)."""
    target_cache = cache or prediction_cache

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(model_name: str, model_version: int | str, features: Any, *args: Any, **kwargs: Any) -> Any:
            cached = target_cache.get(model_name, model_version, features)
            if cached is not None:
                return cached
            result = fn(model_name, model_version, features, *args, **kwargs)
            target_cache.set(model_name, model_version, features, result, ttl_seconds=ttl_seconds)
            return result

        return wrapper

    return decorator
