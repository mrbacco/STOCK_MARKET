#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: runtime_config.py
#############################

"""Environment-backed runtime settings for local and scaled deployments.

The application intentionally keeps its original zero-configuration behavior:
without environment variables it uses local SQLite, Streamlit's process cache,
and one in-process sentiment collector.  The production container stack sets
PostgreSQL, Redis, and worker-mode variables so every web replica becomes
stateless and shares the same durable state.
"""

from __future__ import annotations

import os

# joblib otherwise shells out to count physical cores and prints a traceback
# on Windows when that fails; the logical count is what it falls back to anyway.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def env_bool(name: str, default: bool = False) -> bool:
    """Read a forgiving boolean environment variable."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default)
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
# Upper bound of pooled PostgreSQL connections per process (web replica or worker).
DATABASE_POOL_SIZE = max(int(os.getenv("DATABASE_POOL_SIZE", "5")), 1)
REDIS_URL = os.getenv("REDIS_URL", "").strip()
CACHE_NAMESPACE = os.getenv("CACHE_NAMESPACE", "stock-market").strip() or "stock-market"

# Development remains convenient, but a configured shared database is treated
# as a production signal and does not silently launch a collector in every web
# replica.  Operators can still override this explicitly for a single process.
RUN_IN_PROCESS_SENTIMENT = env_bool(
    "RUN_IN_PROCESS_SENTIMENT",
    default=not bool(DATABASE_URL),
)

# In the production compose stack, analytics workers warm Redis and web replicas
# only read those expensive entries.  Local development computes cache misses
# synchronously so `streamlit run app.py` continues to work by itself.
ANALYTICS_READ_ONLY = env_bool("ANALYTICS_READ_ONLY", default=False)

# The in-process collector loads FinBERT (~10 s of transformers/torch import)
# once it finds new headlines. Waiting before its first cycle keeps that CPU
# burst away from the first page render after `streamlit run app.py`.
SENTIMENT_STARTUP_DELAY_SECONDS = max(int(os.getenv("SENTIMENT_STARTUP_DELAY_SECONDS", "30")), 0)
ANALYTICS_INTERVAL_SECONDS = max(
    int(os.getenv("ANALYTICS_INTERVAL_SECONDS", "900")),
    60,
)
ANALYTICS_PERIODS = tuple(
    value.strip()
    for value in os.getenv("ANALYTICS_PERIODS", "5y").split(",")
    if value.strip()
)
ANALYTICS_HORIZONS = tuple(
    int(value.strip())
    for value in os.getenv("ANALYTICS_HORIZONS", "5,21").split(",")
    if value.strip().isdigit() and int(value.strip()) > 0
)

# Provider controls are deliberately conservative.  Shared caching normally
# prevents most duplicate calls; these values protect upstream public services
# when a cache expires or a worker starts cold.
PROVIDER_MAX_ATTEMPTS = max(int(os.getenv("PROVIDER_MAX_ATTEMPTS", "3")), 1)
PROVIDER_BACKOFF_SECONDS = max(float(os.getenv("PROVIDER_BACKOFF_SECONDS", "0.5")), 0.0)
YAHOO_MIN_INTERVAL_SECONDS = max(float(os.getenv("YAHOO_MIN_INTERVAL_SECONDS", "0.25")), 0.0)
NEWS_MIN_INTERVAL_SECONDS = max(float(os.getenv("NEWS_MIN_INTERVAL_SECONDS", "0.20")), 0.0)
MARKETSTACK_MIN_INTERVAL_SECONDS = max(
    float(os.getenv("MARKETSTACK_MIN_INTERVAL_SECONDS", "0.25")),
    0.0,
)
PROVIDER_CIRCUIT_FAILURES = max(int(os.getenv("PROVIDER_CIRCUIT_FAILURES", "5")), 1)
PROVIDER_CIRCUIT_SECONDS = max(int(os.getenv("PROVIDER_CIRCUIT_SECONDS", "60")), 1)

# Yahoo remains the zero-configuration development source. Marketstack is an
# explicit opt-in because it needs an API key and commercial licensing review.
MARKET_DATA_PROVIDER = (
    os.getenv("MARKET_DATA_PROVIDER", "yfinance").strip().lower() or "yfinance"
)
if MARKET_DATA_PROVIDER not in {"yfinance", "marketstack"}:
    raise ValueError(
        "MARKET_DATA_PROVIDER must be either 'yfinance' or 'marketstack'."
    )
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API_KEY", "").strip()
MARKETSTACK_BASE_URL = (
    os.getenv("MARKETSTACK_BASE_URL", "https://api.marketstack.com/v2").strip()
    or "https://api.marketstack.com/v2"
)
MARKET_DATA_LICENSE_CONFIRMED = env_bool(
    "MARKET_DATA_LICENSE_CONFIRMED",
    default=False,
)

# The free live-chart path polls Yahoo rather than opening a streaming socket.
# Sixty seconds matches the finest useful one-minute bar cadence and avoids
# wasting provider requests. Clamp custom values so an accidental "1" cannot
# hammer the public endpoint or keep a lightweight laptop permanently busy.
# On the first leaderboard load after startup, recent last-known-good snapshots
# are shown immediately while live prices download in the background. Snapshots
# older than this are not used as a preview; 0 disables the preview entirely.
SNAPSHOT_PREVIEW_MAX_AGE_HOURS = max(
    float(os.getenv("SNAPSHOT_PREVIEW_MAX_AGE_HOURS", "12")),
    0.0,
)

LIVE_CHART_REFRESH_SECONDS = min(
    max(int(os.getenv("LIVE_CHART_REFRESH_SECONDS", "60")), 30),
    300,
)
