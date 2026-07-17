<!--
author: mrbacco04@gmail.com
date: 2026-07-12
file: README.md
-->

# Stock Market Intelligence Dashboard

A Streamlit dashboard for monitoring public stock market data, market news, sentiment, and trend projections in one place.

## Project Goals

- Detect current market leaders in an Ireland-focused stock universe and pull their public price data.
- Collect current investing-related news headlines.
- Score headline sentiment to estimate short-term market mood.
- Visualize momentum and trend projections for top performers.
- Provide a manual-refresh real-time workflow for intraday monitoring.

## Current v1 Features

- Public price data via Yahoo Finance.
- Ireland-first automatic detection of the top 10 daily performers from a tracked ISEQ 20 Euronext Dublin universe.
- Optional U.S. large-cap daily-gainers screen through Yahoo Finance.
- FTSE MIB benchmark-index tracking for the Italian market.
- Manual ticker selection by geographical market, with searchable examples and custom Yahoo Finance symbols.
- Separate Overview, Charts, and News views so heavier content loads only when selected.
- Intraday mode with manual refresh controls.
- Historical mode for reliable long-range visualization.
- News aggregation from Google News RSS.
- Finance-specific FinBERT sentiment scoring with an automatic VADER fallback.
- Five-minute background collection with deduplicated SQLite history.
- Point-in-time 24-hour sentiment features and price-only versus sentiment model comparison.
- Top grower ranking using recent performance.
- Interactive charts for close price and feature-based forecasts.
- Walk-forward backtests for the trend projection, including error and baseline metrics.
- Terminal logging with BAC_LOG entries for observability.

## Architecture

- Frontend and app runtime: Streamlit.
- Market data: yfinance.
- News feed parsing: feedparser.
- Data processing: pandas and numpy.
- Forecast model: scikit-learn Ridge regression on technical and lagged sentiment features.
- Visualization: Plotly.
- Sentiment analysis: ProsusAI FinBERT through Transformers, with vaderSentiment fallback.
- Sentiment persistence: SQLite in the ignored `data/` directory.

## Repository Structure

- app.py: Main Streamlit entry point and sidebar workflow.
- app_config.py: Shared constants and runtime configuration.
- app_logging.py: Terminal logging and BAC_LOG helpers.
- ticker_catalog.py: Geographical market presets, ticker examples, suffix rules, and currency labels.
- market_data.py: Price, screener, news, and sentiment data loading.
- forecasting.py: Feature engineering, forecasts, and walk-forward backtests.
- sentiment_analysis.py: Cached FinBERT scoring and VADER fallback.
- sentiment_features.py: Leakage-safe, point-in-time sentiment aggregates.
- sentiment_service.py: RSS ingestion and the in-process background collector.
- sentiment_store.py: SQLite schema, watchlist, news history, and collector status.
- sentiment_worker.py: Standalone continuous collector for 24/7 operation.
- views.py: Overview, Charts, and News rendering.
- tests/test_manual_market_ui.py: Offline Streamlit regression test for the geographical manual-ticker workflow.
- tests/test_sentiment_pipeline.py: Offline persistence, leakage, feature, and promotion tests.
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

The default URL is usually:

- http://localhost:8501

The app starts one background sentiment collector per Streamlit process. It runs every five
minutes while the app process is alive. To keep collecting when the dashboard is not open, run
the standalone worker in a continuously supervised terminal or service:

```bash
python sentiment_worker.py
```

Use `python sentiment_worker.py --once` to test one collection cycle. The worker reads the
bounded watchlist most recently registered by the app.

## How To Use

1. Leave **Ireland: ISEQ 20 leaders** selected to rank the ten strongest latest daily moves in the Ireland-focused universe.
2. Use **View** to select **Overview**, **Charts**, or **News**; select **Charts** to see the price charts and forecast backtests.
3. Select **Italy: FTSE MIB index** to track the Italian benchmark, or **U.S. daily gainers** for the Yahoo Finance U.S. screener.
4. Select **Manual tickers**, choose a geographical market, and then select example securities or type another Yahoo Finance symbol.
5. Choose Real-time Mode for intraday tracking, or disable it for historical mode.
6. Click Refresh now to refresh prices, rankings, and news.
7. Monitor terminal logs for BAC_LOG entries.

## Forecasting Approach

The dashboard uses a feature-based regression model that estimates future returns from recent
momentum, volatility, RSI, price structure, volume behavior, and optional point-in-time sentiment.
The initial sentiment experiment is deliberately limited to daily forecasts of one to five
business sessions.

- It is directional, not predictive in a guaranteed sense.
- It works best as a short-horizon market context tool.
- It should not be used as a sole decision engine for investing.

The dashboard reports a walk-forward backtest for each displayed ticker. It evaluates the same
user-selected forecast horizon using only the preceding 120 observations. The price-plus-sentiment
candidate is evaluated against the otherwise identical price-only model and is promoted only when
its recent walk-forward MAE is lower. It also remains unavailable until at least ten historical
price bars have observable news. Model MAE, MAPE, directional accuracy, and baseline comparisons
do not guarantee future returns.

## Data Sources

- Price and volume: Yahoo Finance endpoints through yfinance.
- Ireland-first leader detection: a tracked ISEQ 20 Euronext Dublin universe, ranked from the latest available Yahoo Finance daily closes.
- Optional U.S. leader detection: Yahoo Finance's predefined `day_gainers` screener for eligible U.S. equities.
- Manual market catalogue: curated examples for major exchanges, with automatic Yahoo Finance suffix handling for custom symbols.
- News headlines: Google News RSS ticker queries.
- Sentiment scoring: FinBERT positive, neutral, and negative probabilities on headline plus summary.
- Historical sentiment: local SQLite records containing publication, first-seen, and scoring timestamps.

## Reliability Notes

- Intraday endpoints can be slower or intermittently unavailable.
- Ireland mode ranks a tracked ISEQ 20 Euronext Dublin universe; it is not a complete ranking of every Irish or European listing.
- U.S. auto-detection is a filtered Yahoo Finance screen for liquid, large-cap daily gainers; it is not
  a complete ranking of every listed U.S. stock.
- Intraday values are the latest returned bar closes; their deltas compare consecutive bars, not
  live ticks or daily changes.
- The app includes fallback behavior and manual refresh flow to reduce lockups.
- For best responsiveness in real-time mode, track a small number of symbols.
- Your ability to buy a listed security depends on your broker account, market access, and personal tax circumstances; this app does not determine investment eligibility.

## Security and Privacy

- This project does not require API keys for current data sources.
- Do not store secrets in source files if new providers are added later.

## Disclaimer

This software is provided for education and research purposes only.
It is not financial advice, trading advice, or portfolio management advice.

## Roadmap Ideas

- Backtesting and model error metrics.
- Advanced indicators such as RSI, MACD, and Bollinger Bands.
- Portfolio risk scoring and drawdown analytics.
- Historical-news provider integration for a longer sentiment baseline immediately after setup.

## Author

- mrbacco04@gmail.com
