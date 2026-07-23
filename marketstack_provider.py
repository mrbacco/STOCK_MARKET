#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: marketstack_provider.py
#############################

"""Small, testable Marketstack adapter for licensed daily price history.

The rest of the application works with normalized OHLCV dataframes. Keeping
provider-specific URLs, parameters, payload validation, and symbol aliases in
this module means a future provider can be added without rewriting forecasting
or Streamlit code.

Important licensing note:
Marketstack's free account is suitable only for technical evaluation. Turning
this adapter on in a paid customer product still requires written confirmation
that the chosen plan covers the exact exchanges, external display, retained
history, and derived forecast outputs used by the product.
"""

from __future__ import annotations

from datetime import date
import json
import os
from typing import Mapping

import pandas as pd
import requests

from app_logging import bac_log_kv


DEFAULT_MARKETSTACK_BASE_URL = "https://api.marketstack.com/v2"
MARKETSTACK_TIMEOUT_SECONDS = 12


def load_symbol_map(raw_value: str | None = None) -> dict[str, str]:
    """Read optional Yahoo-to-Marketstack aliases from a JSON environment value.

    Exchange suffixes are not guaranteed to match across vendors. For example,
    an operator can configure ``{"AAPL": "AAPL", "ENI.MI": "vendor-code"}``
    after the provider confirms its canonical identifier.
    """
    raw = (
        raw_value
        if raw_value is not None
        else os.getenv("MARKETSTACK_SYMBOL_MAP", "").strip()
    )
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as ex:
        raise ValueError("MARKETSTACK_SYMBOL_MAP must be valid JSON.") from ex
    if not isinstance(parsed, dict):
        raise ValueError("MARKETSTACK_SYMBOL_MAP must be a JSON object.")
    return {
        str(source).strip().upper(): str(target).strip()
        for source, target in parsed.items()
        if str(source).strip() and str(target).strip()
    }


def _period_start(period: str, today: date | None = None) -> pd.Timestamp:
    """Translate the app's daily history windows into an inclusive start date."""
    current = pd.Timestamp(today or date.today()).normalize()
    offsets = {
        "1mo": pd.DateOffset(months=1),
        "3mo": pd.DateOffset(months=3),
        "6mo": pd.DateOffset(months=6),
        "1y": pd.DateOffset(years=1),
        "2y": pd.DateOffset(years=2),
        "5y": pd.DateOffset(years=5),
    }
    if period not in offsets:
        raise ValueError(f"Marketstack daily history does not support period {period!r}.")
    return current - offsets[period]


def fetch_marketstack_history(
    ticker: str,
    period: str,
    interval: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    symbol_map: Mapping[str, str] | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch and normalize one Marketstack EOD history response.

    Basic Marketstack plans provide end-of-day data, so intraday requests fail
    explicitly. They must never be silently relabelled as real-time forecasts.
    """
    if interval != "1d":
        raise ValueError(
            "The configured Marketstack EOD provider supports interval='1d' only."
        )

    resolved_key = (api_key or os.getenv("MARKETSTACK_API_KEY", "")).strip()
    if not resolved_key:
        raise RuntimeError(
            "MARKETSTACK_API_KEY is required when MARKET_DATA_PROVIDER=marketstack."
        )

    aliases = dict(symbol_map) if symbol_map is not None else load_symbol_map()
    provider_symbol = aliases.get(ticker.upper(), ticker)
    start = _period_start(period)
    end = pd.Timestamp(date.today()).normalize()
    endpoint = f"{(base_url or DEFAULT_MARKETSTACK_BASE_URL).rstrip('/')}/eod"
    params = {
        "access_key": resolved_key,
        "symbols": provider_symbol,
        "date_from": start.date().isoformat(),
        "date_to": end.date().isoformat(),
        # Two years of business-day history fits comfortably under 1,000 rows.
        "limit": 1000,
        "sort": "ASC",
    }

    bac_log_kv(
        "marketstack.history",
        ticker=ticker,
        provider_symbol=provider_symbol,
        period=period,
        interval=interval,
        date_from=params["date_from"],
        date_to=params["date_to"],
    )
    client = session or requests
    response = client.get(
        endpoint,
        params=params,
        timeout=MARKETSTACK_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        error = payload["error"]
        raise RuntimeError(
            f"Marketstack rejected the request: {error.get('message', error)}"
        )

    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        bac_log_kv(
            "marketstack.history",
            ticker=ticker,
            provider_symbol=provider_symbol,
            status="empty",
        )
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        missing = sorted(required.difference(frame.columns))
        raise RuntimeError(f"Marketstack response omitted required fields: {missing}")

    normalized = frame[
        ["date", "open", "high", "low", "close", "volume"]
    ].rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    normalized["Date"] = (
        pd.to_datetime(normalized["Date"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = (
        normalized.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    bac_log_kv(
        "marketstack.history",
        ticker=ticker,
        provider_symbol=provider_symbol,
        status="success",
        rows=len(normalized),
    )
    return normalized
