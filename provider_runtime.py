#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: provider_runtime.py
#############################

"""Rate limiting, retry/backoff, and circuit breaking for public providers."""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

from app_logging import bac_log_kv
from cache_control import get_redis_client
from runtime_config import (
    PROVIDER_BACKOFF_SECONDS,
    PROVIDER_CIRCUIT_FAILURES,
    PROVIDER_CIRCUIT_SECONDS,
    PROVIDER_MAX_ATTEMPTS,
)

T = TypeVar("T")
_LOCAL_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)
_LOCAL_NEXT_ALLOWED: dict[str, float] = defaultdict(float)
_LOCAL_FAILURES: dict[str, int] = defaultdict(int)
_LOCAL_CIRCUIT_UNTIL: dict[str, float] = defaultdict(float)


class ProviderCircuitOpen(RuntimeError):
    """Raised while a repeatedly failing upstream provider is cooling down."""


def _wait_for_rate_slot(provider: str, minimum_interval: float) -> None:
    """Serialize provider starts locally and, when present, across Redis replicas."""
    if minimum_interval <= 0:
        return
    client = get_redis_client()
    if client is not None:
        try:
            lock = client.lock(
                f"provider-rate:{provider}",
                timeout=max(int(minimum_interval * 10), 5),
                blocking_timeout=10,
            )
            with lock:
                key = f"provider-next:{provider}"
                now = time.time()
                raw_next = client.get(key)
                next_allowed = float(raw_next or 0.0)
                if next_allowed > now:
                    time.sleep(next_allowed - now)
                client.set(key, str(time.time() + minimum_interval), ex=60)
            return
        except Exception as ex:  # pragma: no cover - external outage path
            # Provider access remains protected by the process-local limiter if
            # Redis is restarting. BAC_LOG makes the degraded coordination clear.
            bac_log_kv("provider.rate", provider=provider, status="redis_fallback", error=str(ex))

    with _LOCAL_LOCKS[provider]:
        now = time.monotonic()
        wait_seconds = _LOCAL_NEXT_ALLOWED[provider] - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _LOCAL_NEXT_ALLOWED[provider] = time.monotonic() + minimum_interval


def call_provider(
    provider: str,
    operation: str,
    function: Callable[[], T],
    *,
    minimum_interval: float = 0.0,
    attempts: int = PROVIDER_MAX_ATTEMPTS,
) -> T:
    """Execute one upstream operation with bounded retry and a local circuit."""
    now = time.monotonic()
    if _LOCAL_CIRCUIT_UNTIL[provider] > now:
        remaining = _LOCAL_CIRCUIT_UNTIL[provider] - now
        raise ProviderCircuitOpen(
            f"{provider} circuit is cooling down for {remaining:.1f} more seconds."
        )

    last_error: Exception | None = None
    for attempt in range(1, max(int(attempts), 1) + 1):
        try:
            _wait_for_rate_slot(provider, minimum_interval)
            result = function()
            _LOCAL_FAILURES[provider] = 0
            bac_log_kv(
                "provider.call",
                provider=provider,
                operation=operation,
                attempt=attempt,
                status="success",
            )
            return result
        except Exception as ex:
            last_error = ex
            _LOCAL_FAILURES[provider] += 1
            bac_log_kv(
                "provider.call",
                provider=provider,
                operation=operation,
                attempt=attempt,
                status="failed",
                error=str(ex),
            )
            if _LOCAL_FAILURES[provider] >= PROVIDER_CIRCUIT_FAILURES:
                _LOCAL_CIRCUIT_UNTIL[provider] = time.monotonic() + PROVIDER_CIRCUIT_SECONDS
                break
            if attempt < attempts:
                jitter = random.uniform(0.0, PROVIDER_BACKOFF_SECONDS)
                time.sleep(PROVIDER_BACKOFF_SECONDS * (2 ** (attempt - 1)) + jitter)

    assert last_error is not None
    raise last_error
