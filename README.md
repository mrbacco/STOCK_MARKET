<!--
author: mrbacco04@gmail.com
date: 2026-07-12
file: README.md
-->

# Stock Market Intelligence Dashboard

A Streamlit dashboard for monitoring public stock market data, market news, sentiment, and trend projections in one place.

## Project Goals

- See global markets at a glance: indices, volatility, rates, currencies, commodities, crypto, and
  ten tracked stock universes.
- Rank each universe with a pooled model, and show buy/avoid signals only when out-of-sample
  evidence supports them.
- Study any stock in depth: price, a calibrated projection, the model's view, and news sentiment.
- Measure what following the model would have earned after trading costs.

## Current Features

- **Global markets page:** 39 instruments (world indices, VIX/VXN, US Treasury yields, major
  currencies, commodities, Bitcoin and Ether) with 1D/1W/1M/YTD/1Y changes, 20-day volatility,
  and 3-month sparklines; breadth and median performance of every stock universe; the biggest
  gainers and decliners across all universes. It loads in the background, showing recent saved
  prices first.
- **Ten fixed stock universes (317 stocks):** Dow Jones 30, Nasdaq-100 leaders, Euro Stoxx 50,
  FTSE 100 leaders, DAX 40, CAC 40, FTSE MIB, ISEQ 20, Nikkei 225 leaders, and Hang Seng
  leaders, plus a persistent personal watchlist that accepts any Yahoo Finance symbol.
- **Market ranking page:** the pooled ensemble's ranking of the whole universe, preceded by an
  evidence verdict (supported, tentative, or no demonstrated edge) based on rank IC, its
  t-statistic, and top-10 realized excess return. Signals appear only when evidence is supported.
- **Stock page:** price history with a 50%/80% projection band, the model's rank and band for the
  stock, a daily sentiment trend with recent FinBERT-scored headlines, and optional intraday bars.
- **Portfolio page:** a non-overlapping top-N strategy backtest on the untouched evaluation period
  with a trading-cost slider, a bottom-N control, information ratio, hit rate, drawdown, and
  turnover.
- **News & sentiment page:** 24-hour and 7-day sentiment for every stock in the universe and the
  latest headlines.
- **Model health page:** a multi-year walk-forward test, validation metrics, ensemble weights,
  live scoring of recorded projections, and drift across model runs.
- **Multi-year walk-forward test:** replays the production ensemble over five years, retraining
  every quarter on data available at the time, and reports rank IC with a non-overlapping
  t-statistic, quarterly stability, a top-N backtest after costs, and comparisons with simple
  momentum and reversal rules. When it exists, it drives the evidence verdict.
- A pooled Ridge, Elastic Net, and histogram-gradient-boosting ensemble with market context,
  relative strength, beta, breadth, volatility, liquidity, and point-in-time sentiment features,
  plus an optional LightGBM LambdaRank member that joins only when it improves tuning rank IC.
- Ranking bands scaled by each stock's GARCH excess-return volatility and kept on target by
  adaptive conformal inference.
- Finance-specific FinBERT sentiment with a VADER fallback, collected every five minutes into
  PostgreSQL or local SQLite with leakage-safe point-in-time features.
- Persistent production monitoring and model-run drift, terminal BAC_LOG logging (`LOG_LEVEL`).

## Architecture

- Frontend and app runtime: Streamlit.
- Market data: yfinance for local evaluation, with an optional Marketstack EOD
  adapter for a commercially licensed deployment.
- News feed parsing: feedparser.
- Data processing: pandas and numpy.
- Forecast models: a ticker-level Ridge curve plus a market-wide scikit-learn ensemble (Ridge, Elastic Net, histogram gradient boosting, and logistic direction classifier).
- Exchange sessions: pandas-market-calendars for holidays, early closes, and regular intraday hours.
- Visualization: Plotly.
- Sentiment analysis: ProsusAI FinBERT through Transformers, with vaderSentiment fallback.
- Production persistence: PostgreSQL, with SQLite in `data/` as the zero-configuration local fallback.
- Shared cache and distributed locks: Redis, with bounded in-process TTL caches as L1 (`cache_control.cached_result`).
- Production processes: stateless Streamlit replicas, one sentiment worker, and one analytics precomputation worker.

## Repository Structure

- app.py: Entry point: page config, shared sidebar, sentiment collector, and top navigation.
- app_pages/: One script per page (world_markets, market_ranking, stock, portfolio, news, model_health).
- ui_state.py: Sidebar selection, persistent watchlist, and the per-session market analysis shared by pages.
- ui_components.py: Shared Streamlit renderers (tables, evidence banner, charts, monitoring).
- universe_catalog.py: The ten tracked stock universes with company names.
- market_sources.py: Universe registry (calendar, currency, description) and ticker lookups.
- global_markets.py: Cross-asset and universe snapshots, and the background overview loader.
- model_evidence.py: Out-of-sample evidence verdict that gates ranking signals.
- walk_forward.py, walk_forward_store.py: Multi-year rolling walk-forward test, its background
  runner, and result storage.
- portfolio_backtest.py: Non-overlapping top-N strategy backtest with trading costs.
- app_config.py: Shared modelling constants and exchange-calendar mappings.
- app_logging.py: BAC_LOG helpers on top of the standard logging module (`LOG_LEVEL`).
- market_data.py: Price batches, snapshot recovery, universe leaderboards, and news loading.
- forecasting.py: Feature engineering, ticker projections, and walk-forward backtests.
- chart_pipeline.py: Streamlit-free price loading, ranking, ticker projections, and monitoring records.
- market_model.py: Pooled ensemble, LightGBM ranker, rank IC, probabilities, and bands.
- volatility.py: Point-in-time GARCH(1,1) horizon volatility with an EWMA fallback.
- conformal.py: Volatility-scaled adaptive conformal prediction intervals.
- model_monitoring.py: Persistent forecast outcomes, rolling production metrics, and drift snapshots.
- sentiment_analysis.py, sentiment_features.py, sentiment_service.py, sentiment_store.py:
  FinBERT scoring, point-in-time features, RSS collection, and storage.
- database.py, cache_control.py, provider_runtime.py, runtime_config.py, market_snapshot_store.py:
  persistence, caching, provider protection, settings, and last-known-good price snapshots.
- sentiment_worker.py, analytics_worker.py: Standalone production workers.
- tests/: Offline unit and end-to-end tests (`python -m unittest discover -s tests`).
- requirements.txt / requirements.lock: Direct dependencies and the pinned Linux set for Docker and CI.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/mrbacco/STOCK_MARKET.git
cd STOCK_MARKET
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the direct dependencies. `requirements.lock` pins the
full Linux dependency set (with the CPU-only PyTorch wheel) that the Docker
image and CI install; its header shows how to regenerate it.

Run the offline test suite with:

```bash
python -m unittest discover -s tests
```

### 4. Run the app

```bash
streamlit run app.py
```

For a lighter Windows laptop run without Docker or the continuous FinBERT
collector, use:

```powershell
.\run-local.ps1
```

Forecasting, SQLite persistence, provider retries, and last-known-good market
snapshots remain enabled. Existing sentiment history can still be read. Press
`Ctrl+C` in the PowerShell window to stop the app.

The default URL is usually:

- http://localhost:8501

In zero-configuration local mode, the app starts one background sentiment collector in its
Streamlit process. For a durable or multi-replica deployment, disable that collector and run the
standalone worker in a continuously supervised terminal or service:

```bash
python sentiment_worker.py
```

Use `python sentiment_worker.py --once` to test one collection cycle. The worker reads the
bounded watchlist most recently registered by the app.

### 5. Run the scalable production stack

The included Compose topology starts PostgreSQL, Redis, a dedicated sentiment worker, a dedicated
analytics worker, Streamlit, and an Nginx reverse proxy with sticky WebSocket routing:

```bash
copy .env.example .env
docker compose up --build
```

Open `http://localhost:8501`. To add web capacity without duplicating collectors or model work:

```bash
docker compose up --build --scale app=3
```

Production web replicas set `ANALYTICS_READ_ONLY=true`: rankings, forecast curves, and backtests
are read from worker-warmed Redis entries. PostgreSQL stores sentiment and monitoring history,
while Redis also coordinates provider rate limits, targeted refresh generations, and cache-miss
locks. A period, horizon, or manual portfolio that is not warm yet is placed on a deduplicated
Redis work queue; the UI remains responsive while `analytics_worker.py` prepares it. The
`/_stcore/health` endpoint and proxy `/healthz` route are available to orchestrators.

The default worker warms the common `1y` period and `1,3,5` horizons. Override
`ANALYTICS_PERIODS` or `ANALYTICS_HORIZONS` in `.env` when other combinations should be served
without synchronous computation.

## How To Use

1. **Global markets** opens first: scan indices, rates, currencies, commodities, and every stock
   universe. Select a stock in the movers tables to open it.
2. Pick a **Stock universe** in the sidebar (or **My watchlist** and add any Yahoo Finance
   symbols), a **History window**, and a **Forecast horizon**.
3. **Market ranking** trains the pooled model on the universe. Read the evidence banner first:
   without a demonstrated edge the table shows unproven scores and no signals. Select a row to
   open the stock.
4. **Stock** shows the projection with its bands, the model's view, sentiment, and headlines.
5. **Portfolio** shows what holding the model's top picks would have earned after costs.
6. **News & sentiment** and **Model health** cover sentiment across the universe and model
   validation, live scoring, and drift.
7. **Refresh data** in the sidebar reloads prices, rankings, and the global overview. Set
   `LOG_LEVEL=DEBUG` for detailed terminal logs.

## Forecasting Approach

The dashboard uses two complementary layers. A market-wide ensemble estimates each candidate's
future return relative to the selected market and chooses the top ten automatically. A
ticker-level curve then estimates the future close for each displayed stock. Inputs include
momentum, volatility, RSI, price structure, volume, market breadth, relative strength, beta,
liquidity, and point-in-time financial-news sentiment. Daily forecasts are designed for short
horizons of one to five exchange sessions.

- It is directional, not predictive in a guaranteed sense.
- It works best as a short-horizon market context tool.
- It should not be used as a sole decision engine for investing.

The most reliable verdict comes from the multi-year walk-forward test on the Model health page.
Every quarter of the last five years, the ensemble is rebuilt exactly as in production, using
only data whose outcomes were known at the time (with a forecast-horizon gap), and it predicts
the following quarter. Rank IC significance is measured on non-overlapping dates, and the model
is compared with simple 20-day momentum and 5-day reversal rules. The design follows FinRL's
rolling-window retraining.

The ranking is judged mainly by rank IC: for each evaluation date, the rank correlation between
the predicted and the realized order of stocks, averaged over dates. Values persistently above 0
mean the ranking beats chance. A LightGBM LambdaRank model, trained directly on each day's
ordering, is added to the ensemble only if it raises the tuning period's rank IC above zero and
above the ensemble without it. The ranking's 50% and 80% bands are `prediction +/- q * sigma^gamma`,
where `sigma` is a GARCH(1,1) forecast of the stock's excess-return volatility (estimated only on
the earliest period), `gamma` is learned from tuning errors, and `q` comes from adaptive conformal
inference that learns only from outcomes already realized at each evaluation date.

The dashboard reports walk-forward tests at the selected horizon. Ensemble weights are learned on
an earlier tuning window and measured on a later untouched evaluation window, with a full
forecast-horizon gap between partitions. Sentiment can replace the ticker-level price model only
after at least 20 identical forecast dates are paired and sentiment lowers MAE. Production
forecasts are frozen locally, resolved after their target session, and summarized separately from
historical validation. Model MAE, MAPE, direction, probability, interval coverage, and baseline
comparisons do not guarantee future returns.

## Data Sources

- Prices, indices, rates, currencies, commodities, and crypto: Yahoo Finance through yfinance.
- Stock universes: fixed lists of large index members, checked against Yahoo Finance price
  availability on 8 October 2026 (see `universe_catalog.py`). Index membership changes over time.
- News headlines: Google News RSS queries by company name.
- Sentiment scoring: FinBERT positive, neutral, and negative probabilities on headline plus summary.
- Historical sentiment: PostgreSQL in production or local SQLite, with publication, first-seen,
  and scoring timestamps.

## Reliability Notes

- Intraday endpoints can be slower or intermittently unavailable.
- Universes are tracked lists of large index members, not complete or always-current index replicas.
- The FTSE MIB universe omits STMicroelectronics because Yahoo Finance does not return its Milan listing.
- Intraday bars on the Stock page can be delayed by Yahoo Finance by 15-20 minutes.
- Batch price gaps retry through bounded single-ticker requests. Successful
  histories are stored as last-known-good snapshots in PostgreSQL or local
  SQLite and are clearly labelled when used during provider recovery.
- Severe bar staleness is shown in the Market ranking data-health strip. Stale
  histories are excluded from the ranking, and recovery projections are not
  recorded as fresh production forecasts.
- The Global markets overview needs about 45 Yahoo requests on a cold start; it
  loads in the background and shows recent saved prices first.
- On the first universe leaderboard after startup, recent saved prices
  (up to `SNAPSHOT_PREVIEW_MAX_AGE_HOURS`, default 12) are shown immediately
  with a "saved at" note while live prices download; the page updates itself
  when they arrive. Set the value to `0` to always wait for live prices.
- The in-process sentiment collector waits `SENTIMENT_STARTUP_DELAY_SECONDS`
  (default 30) before its first cycle so loading FinBERT does not slow the first
  page. A cached FinBERT model loads without contacting the Hugging Face Hub.
- Your ability to buy a listed security depends on your broker account, market access, and personal tax circumstances; this app does not determine investment eligibility.

## Security and Privacy

- This project does not require API keys for current data sources.
- Do not store secrets in source files if new providers are added later.

## Disclaimer

This software is provided for education and research purposes only.
It is not financial advice, trading advice, or portfolio management advice.

## Roadmap Ideas

- Licensed historical-news and historical-index-membership data to remove the remaining cold-start and survivorship limitations.
- Sector classifications and macro features such as rates, FX, and index futures.
- Portfolio-level risk, transaction-cost, and drawdown simulation.

## Author

- mrbacco04@gmail.com
