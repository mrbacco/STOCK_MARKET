#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: cache_control.py
#############################

"""Result caching, cache generations, and stampede protection.

`cached_result` gives data and model functions two cache layers without any
Streamlit dependency, so web replicas and standalone workers share one code
path:

- L1 is a bounded in-process TTL cache keyed on the exact argument content.
- L2 is an optional Redis cache shared by replicas and workers. Its keys use a
  *stable* fingerprint: a price history is identified by its completed bars and
  the timestamp of its newest bar, never by pandas ``attrs`` provenance or the
  still-moving prices of a forming bar. A worker-warmed result therefore stays
  addressable when a web replica re-downloads the same history a minute later.

Values are compressed pickles because the cache is private application
infrastructure and must preserve pandas types. Never expose this Redis
instance to untrusted writers.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import pickle
import threading
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Generic, TypeVar, cast

import pandas as pd

from app_logging import bac_debug_kv, bac_log_kv, bac_log_section
from runtime_config import CACHE_NAMESPACE, REDIS_URL

T = TypeVar("T")
_LOCAL_GENERATIONS: dict[str, int] = {}
_LOCAL_GENERATION_LOCK = threading.Lock()
_CURRENT_SCOPE: ContextVar[str] = ContextVar("stock_market_cache_scope", default="default")


class SharedCacheMiss(RuntimeError):
    """Raised when a read-only web replica needs a result not warmed by a worker."""


def set_cache_scope(scope: str) -> None:
    """Attach one market scope to the current Streamlit rerun or worker task."""
    _CURRENT_SCOPE.set(str(scope).strip().lower() or "default")


def current_cache_scope() -> str:
    """Return the current request/task scope without introducing global user state."""
    return _CURRENT_SCOPE.get()


@lru_cache(maxsize=1)
def get_redis_client():
    """Return one process-wide Redis client, or None for local fallback mode."""
    if not REDIS_URL:
        return None
    try:
        import redis

        client = redis.Redis.from_url(
            REDIS_URL,
            socket_connect_timeout=2,
            socket_timeout=3,
            health_check_interval=30,
            decode_responses=False,
        )
        client.ping()
        bac_log_section("cache.redis", "Shared Redis cache is available.")
        return client
    except Exception as ex:  # pragma: no cover - depends on external service
        bac_log_kv("cache.redis", status="unavailable", error=str(ex))
        return None


def _namespaced(key: str) -> str:
    return f"{CACHE_NAMESPACE}:{key}"


def _pandas_digest(value: pd.DataFrame | pd.Series) -> str:
    """Hash pandas content only; ``attrs`` provenance never reaches the key."""
    try:
        payload = pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes()
    except TypeError:
        # Unhashable cells (lists, dicts) fall back to a pickle without attrs.
        stripped = value.copy(deep=False)
        stripped.attrs = {}
        payload = pickle.dumps(stripped, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()


def cache_fingerprint(material: Any, *, stable: bool = False) -> Any:
    """Reduce cache key material to small, deterministic, picklable values.

    ``stable=False`` identifies the exact content. ``stable=True`` identifies a
    dated history (a frame with a ``Date`` column) by its completed rows plus
    the timestamp of its newest row, so a forming bar whose prices move on
    every download does not create a new shared-cache key.
    """
    if isinstance(material, pd.DataFrame):
        columns = tuple(str(column) for column in material.columns)
        if stable and "Date" in material.columns and len(material) >= 2:
            return (
                "dated-frame",
                len(material),
                columns,
                str(material["Date"].iloc[-1]),
                _pandas_digest(material.iloc[:-1]),
            )
        return ("frame", material.shape, columns, _pandas_digest(material))
    if isinstance(material, pd.Series):
        return ("series", len(material), str(material.name), _pandas_digest(material))
    if isinstance(material, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    (str(key), cache_fingerprint(value, stable=stable))
                    for key, value in material.items()
                )
            ),
        )
    if isinstance(material, (list, tuple)):
        return ("sequence", tuple(cache_fingerprint(value, stable=stable) for value in material))
    if isinstance(material, (set, frozenset)):
        return ("set", tuple(sorted(repr(value) for value in material)))
    return material


def _material_digest(material: Any, *, stable: bool = True) -> str:
    fingerprint = cache_fingerprint(material, stable=stable)
    payload = pickle.dumps(fingerprint, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest()


def get_cache_generation(scope: str) -> int:
    """Read a cheap generation number used for targeted cache invalidation."""
    normalized_scope = str(scope).strip().lower()
    client = get_redis_client()
    if client is not None:
        try:
            raw_value = client.get(_namespaced(f"generation:{normalized_scope}"))
            return int(raw_value or 0)
        except Exception as ex:  # pragma: no cover - external outage path
            bac_log_kv("cache.redis", status="generation_read_failed", error=str(ex))
    with _LOCAL_GENERATION_LOCK:
        return _LOCAL_GENERATIONS.get(normalized_scope, 0)


def bump_cache_generation(scope: str) -> int:
    """Invalidate only one logical market/model scope across every replica."""
    normalized_scope = str(scope).strip().lower()
    client = get_redis_client()
    if client is not None:
        try:
            generation = int(client.incr(_namespaced(f"generation:{normalized_scope}")))
            bac_log_kv("cache.generation.bump", scope=normalized_scope, generation=generation)
            return generation
        except Exception as ex:  # pragma: no cover - external outage path
            bac_log_kv("cache.redis", status="generation_write_failed", error=str(ex))
    with _LOCAL_GENERATION_LOCK:
        generation = _LOCAL_GENERATIONS.get(normalized_scope, 0) + 1
        _LOCAL_GENERATIONS[normalized_scope] = generation
    bac_log_kv("cache.generation.bump", scope=normalized_scope, generation=generation)
    return generation


def invalidate_market_scope(scope: str) -> tuple[int, int]:
    """Refresh market inputs and derived analytics without clearing other users."""
    market_generation = bump_cache_generation(f"market:{scope}")
    model_generation = bump_cache_generation(f"model:{scope}")
    return market_generation, model_generation


def enqueue_analytics_job(job_type: str, scope: str, arguments: Any) -> bool:
    """Queue one deduplicated worker request without blocking a web session."""
    client = get_redis_client()
    if client is None:
        return False
    job = {
        "job_type": str(job_type),
        "scope": str(scope),
        "arguments": arguments,
    }
    digest = _material_digest(job)
    dedupe_key = _namespaced(f"analytics-job-pending:{digest}")
    queue_key = _namespaced("analytics-jobs")
    try:
        if int(client.llen(queue_key)) >= 200:
            bac_log_kv("analytics.queue", job_type=job_type, status="queue_full")
            return False
        # A short-lived marker prevents repeated Streamlit reruns from filling
        # the queue with the same large pandas payload while a worker is busy.
        if not client.set(dedupe_key, b"1", nx=True, ex=1800):
            return False
        job["dedupe_key"] = dedupe_key
        payload = zlib.compress(
            pickle.dumps(job, protocol=pickle.HIGHEST_PROTOCOL),
            level=3,
        )
        client.rpush(queue_key, payload)
        bac_log_kv(
            "analytics.queue",
            job_type=job_type,
            scope=scope,
            status="queued",
            bytes=len(payload),
        )
        return True
    except Exception as ex:  # pragma: no cover - external outage path
        bac_log_kv("analytics.queue", job_type=job_type, status="queue_failed", error=str(ex))
        try:
            client.delete(dedupe_key)
        except Exception:
            pass
        return False


def dequeue_analytics_jobs(limit: int = 10) -> list[dict[str, Any]]:
    """Pop a bounded batch for the standalone analytics worker."""
    client = get_redis_client()
    if client is None:
        return []
    jobs: list[dict[str, Any]] = []
    try:
        for _ in range(max(int(limit), 1)):
            payload = client.lpop(_namespaced("analytics-jobs"))
            if payload is None:
                break
            jobs.append(pickle.loads(zlib.decompress(payload)))
    except Exception as ex:  # pragma: no cover - external outage path
        bac_log_kv("analytics.queue", status="dequeue_failed", error=str(ex))
    return jobs


def finish_analytics_job(job: dict[str, Any]) -> None:
    """Release a job's deduplication marker after success or failure."""
    client = get_redis_client()
    dedupe_key = str(job.get("dedupe_key", ""))
    if client is None or not dedupe_key:
        return
    try:
        client.delete(dedupe_key)
    except Exception as ex:  # pragma: no cover - external outage path
        bac_log_kv("analytics.queue", status="finish_failed", error=str(ex))


def shared_cache_get_or_compute(
    namespace: str,
    key_material: Any,
    ttl_seconds: int,
    compute: Callable[[], T],
    *,
    allow_compute: bool = True,
) -> T:
    """Read a cross-process value, computing it once behind a Redis lock.

    With no Redis configured, this deliberately becomes a direct call; the
    `cached_result` decorator keeps its process-local first-level cache.
    """
    client = get_redis_client()
    if client is None:
        if not allow_compute:
            raise SharedCacheMiss(f"No shared cache is configured for {namespace}.")
        return compute()

    digest = _material_digest(key_material)
    cache_key = _namespaced(f"result:{namespace}:{digest}")
    try:
        payload = client.get(cache_key)
    except Exception as ex:  # pragma: no cover - external outage path
        bac_log_kv("cache.redis", namespace=namespace, status="read_failed", error=str(ex))
        if not allow_compute:
            raise SharedCacheMiss(f"Shared cache is unavailable for {namespace}.") from ex
        return compute()
    if payload is not None:
        bac_debug_kv("cache.shared", namespace=namespace, status="hit")
        return pickle.loads(zlib.decompress(payload))

    if not allow_compute:
        bac_debug_kv("cache.shared", namespace=namespace, status="read_only_miss")
        raise SharedCacheMiss(f"The analytics worker has not prepared {namespace} yet.")

    lock = client.lock(
        _namespaced(f"lock:{namespace}:{digest}"),
        timeout=max(int(ttl_seconds), 60),
        blocking_timeout=30,
    )
    try:
        acquired = lock.acquire(blocking=True)
    except Exception as ex:  # pragma: no cover - external outage path
        bac_log_kv("cache.redis", namespace=namespace, status="lock_failed", error=str(ex))
        return compute()
    lock_context = lock if acquired else nullcontext()
    try:
        # Another replica may have populated the key while this process waited.
        try:
            payload = client.get(cache_key)
        except Exception as ex:  # pragma: no cover - external outage path
            bac_log_kv("cache.redis", namespace=namespace, status="recheck_failed", error=str(ex))
            payload = None
        if payload is not None:
            bac_debug_kv("cache.shared", namespace=namespace, status="hit_after_wait")
            return pickle.loads(zlib.decompress(payload))

        result = compute()
        serialized = zlib.compress(
            pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL),
            level=3,
        )
        try:
            client.setex(cache_key, max(int(ttl_seconds), 1), serialized)
            bac_debug_kv(
                "cache.shared",
                namespace=namespace,
                status="stored",
                bytes=len(serialized),
            )
        except Exception as ex:  # pragma: no cover - external outage path
            # The user still receives the computed result; only cross-replica
            # reuse is temporarily lost while Redis recovers.
            bac_log_kv("cache.redis", namespace=namespace, status="write_failed", error=str(ex))
        return result
    finally:
        if acquired:
            try:
                lock_context.release()
            except Exception:
                # Redis lock expiry is harmless here: the computed value was
                # already written and later readers still receive it.
                pass


class _LocalResultCache:
    """Bounded, thread-safe, process-local TTL cache.

    Values are stored pickled so every caller receives an independent copy,
    matching `st.cache_data` semantics: mutating a returned DataFrame (for
    example its ``attrs``) can never corrupt the cached value.
    """

    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self._max_entries = max(int(max_entries), 1)
        self._ttl_seconds = max(float(ttl_seconds), 0.0)
        self._entries: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[bool, Any]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False, None
            expires_at, payload = entry
            if expires_at < time.monotonic():
                del self._entries[key]
                return False, None
            self._entries.move_to_end(key)
        return True, pickle.loads(payload)

    def set(self, key: str, value: Any) -> None:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with self._lock:
            self._entries[key] = (time.monotonic() + self._ttl_seconds, payload)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class CachedFunction(Generic[T]):
    """Type of a `cached_result` function: callable, with `cache_clear()`."""

    def __call__(self, *args: Any, **kwargs: Any) -> T: ...

    def cache_clear(self) -> None: ...


def cached_result(
    namespace: str,
    *,
    ttl_seconds: int,
    max_entries: int,
    generation: str | None = None,
    shared: bool = True,
    allow_compute: bool = True,
    on_failure: Callable[[], Any] | None = None,
) -> Callable[[Callable[..., T]], CachedFunction[T]]:
    """Cache a function in-process (L1) and, optionally, in Redis (L2).

    - ``generation`` ("market" or "model") adds the current scope's generation
      to every key, so "Refresh now" invalidates only that market scope.
    - ``allow_compute=False`` makes a read-only web replica raise
      `SharedCacheMiss` instead of computing an expensive result.
    - ``on_failure`` turns a failure into a degraded value. A shared-cache miss
      additionally queues the call for the analytics worker, using the
      namespace as the job type and the bound arguments as its keywords.
      Without ``on_failure``, every exception propagates to the caller.
    """

    def decorate(function: Callable[..., T]) -> CachedFunction[T]:
        signature = inspect.signature(function)
        local_cache = _LocalResultCache(max_entries, ttl_seconds)

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            scope = current_cache_scope()
            cache_generation = (
                get_cache_generation(f"{generation}:{scope}") if generation else 0
            )
            local_key = _material_digest(
                (namespace, arguments, cache_generation),
                stable=False,
            )
            hit, cached_value = local_cache.get(local_key)
            if hit:
                return cached_value

            def compute() -> T:
                return function(*bound.args, **bound.kwargs)

            try:
                value = (
                    shared_cache_get_or_compute(
                        namespace,
                        (arguments, cache_generation),
                        ttl_seconds,
                        compute,
                        allow_compute=allow_compute,
                    )
                    if shared
                    else compute()
                )
            except SharedCacheMiss as ex:
                if on_failure is None:
                    raise
                enqueue_analytics_job(namespace, scope, arguments)
                bac_log_kv(f"cache.{namespace}", status="worker_queued", error=str(ex))
                return on_failure()
            except Exception as ex:
                if on_failure is None:
                    raise
                # One bad ticker, cache entry, or estimator fit must not take
                # down the page; the caller renders the degraded value.
                bac_log_kv(
                    f"cache.{namespace}",
                    status="failed",
                    error_type=type(ex).__name__,
                    error=str(ex),
                )
                return on_failure()

            local_cache.set(local_key, value)
            return value

        wrapper.cache_clear = local_cache.clear  # type: ignore[attr-defined]
        return cast(CachedFunction[T], wrapper)

    return decorate
