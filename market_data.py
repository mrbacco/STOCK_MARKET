#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: market_data.py
#############################

"""Market-data, ticker parsing, and news-loading helpers.

This module isolates the public-data integration points. That makes it easier
to debug API behavior, cache usage, and malformed market payloads without
mixing those concerns into the Streamlit layout code.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from app_config import (
    AUTO_DETECTED_PERFORMERS,
    FTSE_MIB_NAME,
    FTSE_MIB_TICKER,
    ISEQ_20_DUBLIN_LISTINGS,
    MOMENTUM_PERIODS,
    US_SCREENER_QUERY,
)
from app_logging import bac_log_kv, bac_log_list_preview, bac_log_section
from sentiment_service import collect_tickers_once
from sentiment_store import load_sentiment_history


def parse_tickers(raw: str) -> List[str]:
    """Normalize comma-separated tickers and preserve the user's original order."""
    bac_log_kv("market_data.parse_tickers", raw_input=raw)

    parts = [part.strip().upper() for part in raw.split(",") if part.strip()]

    # Duplicate removal happens after normalization so "aapl" and "AAPL" collapse.
    seen = set()
    clean_tickers: list[str] = []
    for ticker in parts:
        if ticker not in seen:
            seen.add(ticker)
            clean_tickers.append(ticker)

    bac_log_list_preview(
        "market_data.parse_tickers",
        "normalized_tickers",
        clean_tickers,
    )
    return clean_tickers


def format_price_history(history: pd.DataFrame) -> pd.DataFrame:
    """Standardize yfinance history output to the columns the app expects."""
    bac_log_kv(
        "market_data.format_price_history",
        incoming_rows=len(history),
        incoming_columns=list(history.columns),
    )

    if history.empty:
        bac_log_section("market_data.format_price_history", "History frame was empty.")
        return pd.DataFrame()

    history = history.reset_index()
    date_column = "Datetime" if "Datetime" in history.columns else "Date"
    required_columns = {date_column, "Open", "High", "Low", "Close", "Volume"}
    if not required_columns.issubset(history.columns):
        bac_log_kv(
            "market_data.format_price_history",
            missing_columns=sorted(required_columns.difference(history.columns)),
        )
        return pd.DataFrame()

    # The app uses one normalized "Date" column for both daily and intraday data.
    formatted = history[[date_column, "Open", "High", "Low", "Close", "Volume"]].rename(
        columns={date_column: "Date"}
    )
    formatted["Date"] = pd.to_datetime(formatted["Date"]).dt.tz_localize(None)

    bac_log_kv(
        "market_data.format_price_history",
        outgoing_rows=len(formatted),
        outgoing_columns=list(formatted.columns),
    )
    return formatted


@st.cache_data(ttl=60)
def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch history for one ticker.

    This function remains available for targeted debugging and ad-hoc expansion,
    even though the current UI primarily uses the batch loader.
    """
    bac_log_kv(
        "market_data.get_price_history",
        ticker=ticker,
        period=period,
        interval=interval,
    )
    history = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    formatted = format_price_history(history)
    bac_log_kv(
        "market_data.get_price_history",
        ticker=ticker,
        formatted_rows=len(formatted),
    )
    return formatted


@st.cache_data(ttl=30, max_entries=100)
def get_price_history_batch(
    tickers: List[str],
    period: str,
    interval: str,
) -> dict[str, pd.DataFrame]:
    """Fetch many ticker histories in one request to reduce repeated network work."""
    bac_log_list_preview("market_data.get_price_history_batch", "requested_tickers", tickers)
    bac_log_kv(
        "market_data.get_price_history_batch",
        period=period,
        interval=interval,
    )

    result = {ticker: pd.DataFrame() for ticker in tickers}
    if not tickers:
        bac_log_section("market_data.get_price_history_batch", "No tickers were provided.")
        return result

    try:
        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=False,
            progress=False,
            timeout=4,
        )
    except Exception as ex:
        bac_log_kv("market_data.get_price_history_batch", download_error=str(ex))
        return result

    if data is None or data.empty:
        bac_log_section("market_data.get_price_history_batch", "Download returned no rows.")
        return result

    bac_log_kv(
        "market_data.get_price_history_batch",
        raw_shape=data.shape,
        raw_is_multiindex=isinstance(data.columns, pd.MultiIndex),
    )

    # yfinance can return either ticker-first or field-first MultiIndex layouts.
    if isinstance(data.columns, pd.MultiIndex):
        first_level = set(data.columns.get_level_values(0))
        second_level = set(data.columns.get_level_values(1))
        for ticker in tickers:
            if ticker in first_level:
                ticker_data = data[ticker].copy()
            elif ticker in second_level:
                ticker_data = data.xs(ticker, axis=1, level=1).copy()
            else:
                bac_log_kv(
                    "market_data.get_price_history_batch",
                    ticker=ticker,
                    message="Ticker was missing from the MultiIndex payload.",
                )
                continue

            ticker_frame = ticker_data.to_frame().T if isinstance(ticker_data, pd.Series) else ticker_data
            formatted = format_price_history(ticker_frame.dropna(how="all"))
            result[ticker] = formatted
            bac_log_kv(
                "market_data.get_price_history_batch",
                ticker=ticker,
                formatted_rows=len(formatted),
            )
    else:
        formatted = format_price_history(data.copy().dropna(how="all"))
        result[tickers[0]] = formatted
        bac_log_kv(
            "market_data.get_price_history_batch",
            ticker=tickers[0],
            formatted_rows=len(formatted),
        )

    non_empty_tickers = [ticker for ticker, frame in result.items() if not frame.empty]
    bac_log_list_preview(
        "market_data.get_price_history_batch",
        "non_empty_tickers",
        non_empty_tickers,
    )
    return result


@st.cache_data(ttl=60, max_entries=2)
def get_us_top_performers(limit: int = AUTO_DETECTED_PERFORMERS) -> pd.DataFrame:
    """Load the Yahoo Finance U.S. daily gainers screener and keep top equities."""
    bac_log_kv("market_data.get_us_top_performers", limit=limit)
    columns = ["Ticker", "Company", "Daily change", "Last price"]

    try:
        response = yf.screen(US_SCREENER_QUERY, count=max(limit * 3, 30))
    except Exception as ex:
        bac_log_kv("market_data.get_us_top_performers", screener_error=str(ex))
        return pd.DataFrame(columns=columns)

    quotes = response.get("quotes", [])
    bac_log_kv("market_data.get_us_top_performers", quote_count=len(quotes))

    rows = []
    seen_tickers = set()
    for quote in quotes:
        if quote.get("quoteType") != "EQUITY":
            continue

        ticker = str(quote.get("symbol", "")).upper()
        if not ticker or ticker in seen_tickers:
            continue

        try:
            daily_change = float(quote.get("regularMarketChangePercent"))
        except (TypeError, ValueError):
            bac_log_kv(
                "market_data.get_us_top_performers",
                ticker=ticker,
                message="Skipped because daily change could not be parsed.",
            )
            continue

        if not np.isfinite(daily_change):
            bac_log_kv(
                "market_data.get_us_top_performers",
                ticker=ticker,
                message="Skipped because daily change was not finite.",
            )
            continue

        try:
            last_price = float(quote.get("regularMarketPrice"))
        except (TypeError, ValueError):
            last_price = np.nan

        company = (
            quote.get("longName")
            or quote.get("shortName")
            or quote.get("displayName")
            or ticker
        )
        rows.append(
            {
                "Ticker": ticker,
                "Company": company,
                "Daily change": daily_change,
                "Last price": last_price,
            }
        )
        seen_tickers.add(ticker)

        if len(rows) >= limit:
            break

    result = (
        pd.DataFrame(rows).sort_values("Daily change", ascending=False).head(limit).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=columns)
    )
    bac_log_kv("market_data.get_us_top_performers", result_rows=len(result))
    return result


@st.cache_data(ttl=300, max_entries=2)
def get_iseq20_top_performers(limit: int = AUTO_DETECTED_PERFORMERS) -> pd.DataFrame:
    """Rank the fixed Ireland universe by the latest daily change."""
    bac_log_kv("market_data.get_iseq20_top_performers", limit=limit)
    columns = ["Ticker", "Company", "Daily change", "Last price", "Last session"]
    price_data = get_price_history_batch(
        list(ISEQ_20_DUBLIN_LISTINGS),
        period="5d",
        interval="1d",
    )

    rows = []
    for ticker, company in ISEQ_20_DUBLIN_LISTINGS.items():
        history = price_data.get(ticker, pd.DataFrame())
        if history.empty:
            bac_log_kv(
                "market_data.get_iseq20_top_performers",
                ticker=ticker,
                message="Skipped because no history was returned.",
            )
            continue

        closes = history[["Date", "Close"]].dropna().sort_values("Date")
        if len(closes) < 2:
            bac_log_kv(
                "market_data.get_iseq20_top_performers",
                ticker=ticker,
                message="Skipped because fewer than two closes were available.",
            )
            continue

        previous_close = float(closes["Close"].iloc[-2])
        last_price = float(closes["Close"].iloc[-1])
        if previous_close == 0:
            bac_log_kv(
                "market_data.get_iseq20_top_performers",
                ticker=ticker,
                message="Skipped because the previous close was zero.",
            )
            continue

        daily_change = ((last_price - previous_close) / previous_close) * 100
        if not np.isfinite(daily_change):
            bac_log_kv(
                "market_data.get_iseq20_top_performers",
                ticker=ticker,
                message="Skipped because the daily change was not finite.",
            )
            continue

        rows.append(
            {
                "Ticker": ticker,
                "Company": company,
                "Daily change": daily_change,
                "Last price": last_price,
                "Last session": pd.Timestamp(closes["Date"].iloc[-1]).date(),
            }
        )

    result = (
        pd.DataFrame(rows).sort_values("Daily change", ascending=False).head(limit).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=columns)
    )
    bac_log_kv("market_data.get_iseq20_top_performers", result_rows=len(result))
    return result


@st.cache_data(ttl=300, max_entries=2)
def get_ftse_mib_index() -> pd.DataFrame:
    """Load the FTSE MIB index as a single tracked market source."""
    bac_log_section("market_data.get_ftse_mib_index", "Loading FTSE MIB index history.")
    columns = ["Ticker", "Company", "Daily change", "Last price", "Last session"]
    price_data = get_price_history_batch([FTSE_MIB_TICKER], period="5d", interval="1d")
    history = price_data.get(FTSE_MIB_TICKER, pd.DataFrame())

    if history.empty:
        bac_log_section("market_data.get_ftse_mib_index", "No FTSE MIB history was returned.")
        return pd.DataFrame(columns=columns)

    closes = history[["Date", "Close"]].dropna().sort_values("Date")
    if len(closes) < 2:
        bac_log_kv(
            "market_data.get_ftse_mib_index",
            message="Fewer than two close values were returned.",
            close_rows=len(closes),
        )
        return pd.DataFrame(columns=columns)

    previous_close = float(closes["Close"].iloc[-2])
    last_price = float(closes["Close"].iloc[-1])
    if previous_close == 0:
        bac_log_section("market_data.get_ftse_mib_index", "Previous FTSE MIB close was zero.")
        return pd.DataFrame(columns=columns)

    daily_change = ((last_price - previous_close) / previous_close) * 100
    if not np.isfinite(daily_change):
        bac_log_kv(
            "market_data.get_ftse_mib_index",
            message="Daily change was not finite.",
            daily_change=daily_change,
        )
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(
        [
            {
                "Ticker": FTSE_MIB_TICKER,
                "Company": FTSE_MIB_NAME,
                "Daily change": daily_change,
                "Last price": last_price,
                "Last session": pd.Timestamp(closes["Date"].iloc[-1]).date(),
            }
        ]
    )
    bac_log_kv(
        "market_data.get_ftse_mib_index",
        daily_change=daily_change,
        last_price=last_price,
        last_session=str(result["Last session"].iloc[0]),
    )
    return result


@st.cache_data(ttl=60, max_entries=200)
def get_news(ticker: str, company_name: str = "", max_items: int = 20) -> pd.DataFrame:
    """Collect current headlines and return persistent financial sentiment."""
    bac_log_kv(
        "market_data.get_news",
        ticker=ticker,
        company_name=company_name,
        max_items=max_items,
    )
    collect_tickers_once({ticker: company_name or ticker})
    result = load_sentiment_history(ticker, limit=max_items)
    if result.empty:
        bac_log_kv("market_data.get_news", ticker=ticker, result_rows=0)
        return pd.DataFrame()
    result = result.rename(columns={"published_at": "published"})
    bac_log_kv("market_data.get_news", ticker=ticker, result_rows=len(result))
    return result


def growth_score(df: pd.DataFrame, periods: int = MOMENTUM_PERIODS) -> float:
    """Compute a simple percentage-change score over the requested lookback window."""
    bac_log_kv("market_data.growth_score", rows=len(df), periods=periods)
    if df.empty or len(df) < periods + 1:
        bac_log_section("market_data.growth_score", "Insufficient history for momentum score.")
        return float("-inf")

    start = df["Close"].iloc[-(periods + 1)]
    end = df["Close"].iloc[-1]
    if start == 0:
        bac_log_section("market_data.growth_score", "Start price was zero; returning -inf.")
        return float("-inf")

    score = ((end - start) / start) * 100.0
    bac_log_kv("market_data.growth_score", start=float(start), end=float(end), score=score)
    return score


def momentum_label(realtime_mode: bool, interval: str) -> str:
    """Describe the momentum lookback in words that match the current mode."""
    label = f"{MOMENTUM_PERIODS}-bar ({interval})" if realtime_mode else f"{MOMENTUM_PERIODS}-session"
    bac_log_kv(
        "market_data.momentum_label",
        realtime_mode=realtime_mode,
        interval=interval,
        label=label,
    )
    return label


def load_news_frames_parallel(
    tickers: List[str],
    company_by_ticker: dict[str, str],
) -> list[pd.DataFrame]:
    """Fetch headline frames in parallel so the News view stays responsive."""
    bac_log_list_preview("market_data.load_news_frames_parallel", "tickers", tickers)
    if not tickers:
        bac_log_section("market_data.load_news_frames_parallel", "No tickers were supplied.")
        return []

    news_frames: list[pd.DataFrame] = []
    max_workers = min(4, len(tickers))
    bac_log_kv("market_data.load_news_frames_parallel", max_workers=max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_news, ticker, company_by_ticker.get(ticker, "")): ticker
            for ticker in tickers
        }
        bac_log_kv(
            "market_data.load_news_frames_parallel",
            submitted_jobs=len(futures),
        )
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                news_df = future.result()
            except Exception as ex:
                bac_log_kv(
                    "market_data.load_news_frames_parallel",
                    ticker=ticker,
                    news_error=str(ex),
                )
                continue

            bac_log_kv(
                "market_data.load_news_frames_parallel",
                ticker=ticker,
                fetched_rows=len(news_df),
            )
            if not news_df.empty:
                news_frames.append(news_df)

    bac_log_kv(
        "market_data.load_news_frames_parallel",
        non_empty_frames=len(news_frames),
    )
    return news_frames
