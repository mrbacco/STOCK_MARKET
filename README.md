<!--
author: mrbacco04@gmail.com
date: 2026-07-12
file: README.md
-->

# Stock Market Intelligence Dashboard

A Streamlit dashboard for monitoring public stock market data, market news, sentiment, and trend projections in one place.

## Project Goals

- Detect current market leaders and pull their public price data.
- Collect current investing-related news headlines.
- Score headline sentiment to estimate short-term market mood.
- Visualize momentum and trend projections for top performers.
- Provide a manual-refresh real-time workflow for intraday monitoring.

## Current v1 Features

- Public price data via Yahoo Finance.
- Automatic detection of the top 10 eligible U.S. daily gainers through Yahoo Finance's screener.
- Intraday mode with manual refresh controls.
- Historical mode for reliable long-range visualization.
- News aggregation from Google News RSS.
- VADER sentiment scoring for each news item.
- Top grower ranking using recent performance.
- Interactive charts for close price and forecast trend.
- Walk-forward backtests for the trend projection, including error and baseline metrics.
- Terminal logging with BAC_LOG entries for observability.

## Architecture

- Frontend and app runtime: Streamlit.
- Market data: yfinance.
- News feed parsing: feedparser.
- Data processing: pandas and numpy.
- Forecast model: scikit-learn LinearRegression.
- Visualization: Plotly.
- Sentiment analysis: vaderSentiment.

## Repository Structure

- app.py: Main Streamlit application.
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

## How To Use

1. Leave **Auto-detect top gainers** selected to chart the ten currently detected market leaders.
2. Select **Manual tickers** if you want to enter specific symbols instead.
3. Choose Real-time Mode for intraday tracking, or disable it for historical mode.
4. Click Refresh now to refresh the screener, prices, and news.
5. Review the detected-gainers table, chart overlays, and news sentiment sections.
6. Monitor terminal logs for BAC_LOG entries.

## Forecasting Approach

The dashboard uses a lightweight linear regression model over recent closing prices.

- It is directional, not predictive in a guaranteed sense.
- It works best as a quick trend indicator.
- It should not be used as a sole decision engine for investing.

The dashboard also reports a walk-forward backtest for each displayed ticker. It makes up to 30
one-bar forecasts using only the preceding 60 observations, then compares those unseen outcomes
with a no-change baseline. Model MAE, MAPE, directional accuracy, and MAE improvement versus the
baseline show how the current trend model has recently performed. They do not guarantee future
returns.

## Data Sources

- Price and volume: Yahoo Finance endpoints through yfinance.
- Top-performer detection: Yahoo Finance's predefined `day_gainers` screener for eligible U.S. equities.
- News headlines: Google News RSS ticker queries.
- Sentiment scoring: VADER compound score on headline plus summary.

## Reliability Notes

- Intraday endpoints can be slower or intermittently unavailable.
- Auto-detection is a filtered Yahoo Finance screen for liquid, large-cap U.S. daily gainers; it is not
  a complete ranking of every listed stock.
- Intraday values are the latest returned bar closes; their deltas compare consecutive bars, not
  live ticks or daily changes.
- The app includes fallback behavior and manual refresh flow to reduce lockups.
- For best responsiveness in real-time mode, track a small number of symbols.

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
- Non-blocking background refresh and async data ingestion.

## Author

- mrbacco04@gmail.com
