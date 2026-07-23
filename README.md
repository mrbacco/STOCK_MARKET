<!--
author: mrbacco04@gmail.com
date: 2026-07-12
file: README.md
-->

# Stock Market Intelligence Dashboard

A Streamlit dashboard for monitoring public stock market data, market news, sentiment, and trend projections in one place.

## Project Goals

- Evaluate a broad market candidate pool and chart the ten strongest forward predictions automatically.
- Collect current investing-related news headlines.
- Score headline sentiment to estimate short-term market mood.
- Visualize momentum and trend projections for top performers.
- Provide a manual-refresh real-time workflow for intraday monitoring.

## Current Features

- Public price data via Yahoo Finance.
- Automatic market-wide top-10 prediction ranking for tracked Ireland, Italy, and U.S. sources.
- Ireland ranking from a tracked ISEQ 20 Euronext Dublin universe.
- Italy ranking across 39 Yahoo-supported FTSE MIB constituents.
- U.S. large-cap daily-gainers ranking through Yahoo Finance.
- Manual ticker selection by geographical market, with searchable examples and custom Yahoo Finance symbols.
- Separate Overview, Charts, and News views so heavier content loads only when selected.
- Intraday mode with manual refresh controls.
- Historical mode for reliable long-range visualization.
- News aggregation from Google News RSS.
- Finance-specific FinBERT sentiment scoring with an automatic VADER fallback.
- Five-minute worker collection with deduplicated PostgreSQL or local SQLite history.
- Point-in-time sentiment with exchange-close cutoffs, recency decay, source quality, relevance, novelty, event intensity, negative share, and volume shocks.
- A pooled Ridge, Elastic Net, and histogram-gradient-boosting ensemble with market context, relative strength, beta, breadth, volatility, and liquidity features.
- Horizon-embargoed tuning/evaluation periods, paired sentiment promotion, calibrated outperformance probability, model agreement, and abstention signals.
- Forecast 50% and 80% uncertainty intervals plus exchange-aware future sessions and intraday bars.
- Persistent production monitoring for rolling MAE, return MAE, directional accuracy, interval coverage, volatility regime, and model-run drift.
- Top grower ranking using recent performance.
- Interactive charts for close price and feature-based forecasts.
- Walk-forward backtests for the trend projection, including error and baseline metrics.
- Terminal logging with BAC_LOG entries for observability.

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
- Shared cache and distributed locks: Redis, with bounded Streamlit process caches as L1.
- Production processes: stateless Streamlit replicas, one sentiment worker, and one analytics precomputation worker.

## Repository Structure

- app.py: Main Streamlit entry point and sidebar workflow.
- app_config.py: Shared constants and runtime configuration.
- app_logging.py: Terminal logging and BAC_LOG helpers.
- ticker_catalog.py: Geographical market presets, ticker examples, suffix rules, and currency labels.
- market_data.py: Price, screener, news, and sentiment data loading.
- forecasting.py: Feature engineering, forecasts, and walk-forward backtests.
- market_model.py: Pooled contextual ensemble, probabilities, intervals, and automatic top-10 ranking.
- model_monitoring.py: Persistent forecast outcomes, rolling production metrics, and drift snapshots.
- sentiment_analysis.py: Cached FinBERT scoring and VADER fallback.
- sentiment_features.py: Leakage-safe, point-in-time sentiment aggregates.
- sentiment_service.py: RSS ingestion and the in-process background collector.
- database.py: PostgreSQL/SQLite DB-API compatibility layer.
- cache_control.py: Redis result caching, cache generations, and stampede locks.
- provider_runtime.py: provider rate limiting, retry/backoff, and circuit breaking.
- runtime_config.py: environment-backed local and production runtime settings.
- sentiment_store.py: portable schema, watchlist, news history, and collector status.
- sentiment_worker.py: Standalone continuous collector for 24/7 operation.
- analytics_worker.py: Standalone market-ranking and backtest precomputation worker.
- views.py: Overview, Charts, and News rendering.
- tests/test_manual_market_ui.py: Offline Streamlit regression test for the geographical manual-ticker workflow.
- tests/test_market_leader_rankings.py: Offline ranking and ten-ticker-cap tests for automatic market sources.
- tests/test_sentiment_pipeline.py: Offline persistence, leakage, feature, and promotion tests.
- tests/test_market_model.py: Pooled ensemble, probability, interval, and embargo tests.
- tests/test_forecast_calendar.py: Exchange-session and holiday projection tests.
- tests/test_model_monitoring.py: Forecast-resolution and drift-monitoring tests.
- tests/test_prediction_ranking_ui.py: Offline Streamlit proof that the model-ranked top ten charts render automatically.
- requirements.txt: Python dependencies.
- README.md: Project documentation.
- LICENSE: MIT license.

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

1. Choose **Ireland: ISEQ 20 leaders**, **Italy: FTSE MIB leaders**, or **U.S. daily gainers** to rank that source automatically.
2. Use **View** to select **Overview**, **Charts**, or **News**; select **Charts** to see the price charts and forecast backtests.
3. The Charts view automatically loads the highest-ranked ten supported tickers for the selected automatic market source.
4. Select **Manual tickers**, choose a geographical market, and then select example securities or type another Yahoo Finance symbol.
5. Choose Real-time Mode for intraday tracking. **Live chart updates** then
   refreshes the Charts view every 60 seconds using free Yahoo polling while
   the browser tab remains active.
6. Disable Live chart updates for manual-only operation, or click Refresh now
   to invalidate prices, rankings, and forecasts immediately.
7. Monitor terminal logs for BAC_LOG entries.

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

The dashboard reports walk-forward tests at the selected horizon. Ensemble weights are learned on
an earlier tuning window and measured on a later untouched evaluation window, with a full
forecast-horizon gap between partitions. Sentiment can replace the ticker-level price model only
after at least 20 identical forecast dates are paired and sentiment lowers MAE. Production
forecasts are frozen locally, resolved after their target session, and summarized separately from
historical validation. Model MAE, MAPE, direction, probability, interval coverage, and baseline
comparisons do not guarantee future returns.

## Data Sources

- Price and volume: Yahoo Finance endpoints through yfinance.
- Ireland-first leader detection: a tracked ISEQ 20 Euronext Dublin universe, ranked from the latest available Yahoo Finance daily closes.
- Italy leader detection: 39 current FTSE MIB constituents with Yahoo-supported Milan history, ranked from the latest available daily closes.
- Optional U.S. leader detection: Yahoo Finance's predefined `day_gainers` screener for eligible U.S. equities.
- Manual market catalogue: curated examples for major exchanges, with automatic Yahoo Finance suffix handling for custom symbols.
- News headlines: Google News RSS ticker queries.
- Sentiment scoring: FinBERT positive, neutral, and negative probabilities on headline plus summary.
- Historical sentiment: PostgreSQL in production or local SQLite records, containing publication, first-seen, and scoring timestamps.

## Reliability Notes

- Intraday endpoints can be slower or intermittently unavailable.
- Ireland mode ranks a tracked ISEQ 20 Euronext Dublin universe; it is not a complete ranking of every Irish or European listing.
- Italy mode ranks 39 FTSE MIB constituents. STMicroelectronics is omitted because Yahoo Finance does not currently return its Milan listing.
- U.S. auto-detection is a filtered Yahoo Finance screen for liquid, large-cap daily gainers; it is not
  a complete ranking of every listed U.S. stock.
- Intraday values are the latest returned bar closes; their deltas compare consecutive bars, not
  live ticks or daily changes.
- Free live charts poll once per minute; they are not exchange tick streams.
  Plot zoom and legend choices remain stable across refreshes, while forecast
  inputs advance only after a five-minute bucket completes to control CPU use.
- Batch price gaps retry through bounded single-ticker requests. Successful
  histories are stored as last-known-good snapshots in PostgreSQL or local
  SQLite and are clearly labelled when used during provider recovery.
- Severe bar staleness is shown in the Charts data-health strip. Recovery
  forecasts remain visible but are not recorded as fresh production forecasts.
- For best responsiveness in real-time mode, track a small number of symbols.
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
