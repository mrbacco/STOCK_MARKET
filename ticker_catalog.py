#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: ticker_catalog.py
#############################

"""Curated geographical market presets for the manual ticker workflow.

The catalogue is deliberately separate from the Streamlit entry point:
- the sidebar can stay focused on rendering widgets
- Yahoo Finance exchange suffix rules live in one place
- currency labels stay aligned with the selected market
- the preset list can grow without making `app.py` difficult to scan

The securities below are examples, not a complete exchange directory. Users
can always type another Yahoo Finance symbol into the Streamlit multiselect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app_logging import bac_log_kv, bac_log_list_preview, bac_log_section


@dataclass(frozen=True)
class ManualMarketPreset:
    """Describe one geographical market and its manual-ticker behavior."""

    # `key` is a short, stable value used in dynamic Streamlit widget keys.
    key: str
    # `label` is the human-readable option shown in the market dropdown.
    label: str
    # `exchange` gives users enough context to understand the selected venue.
    exchange: str
    # Yahoo Finance uses exchange suffixes such as `.DE`, `.L`, and `.T`.
    yahoo_suffix: str
    # The three display fields keep metrics, tables, and chart axes consistent.
    currency_prefix: str
    price_format: str
    price_axis_label: str
    # A short description appears directly below the market selector.
    description: str
    # Each pair is `(Yahoo Finance ticker, company or index name)`.
    ticker_examples: tuple[tuple[str, str], ...]

    @property
    def ticker_symbols(self) -> tuple[str, ...]:
        """Return only the symbols, preserving the curated display order."""
        return tuple(ticker for ticker, _company in self.ticker_examples)

    def company_name(self, ticker: str) -> str:
        """Return a known company name, or the ticker for a custom symbol."""
        ticker_upper = ticker.upper()
        for symbol, company in self.ticker_examples:
            if symbol.upper() == ticker_upper:
                return company
        return ticker_upper

    def price_display(self) -> tuple[str, str, str]:
        """Return the prefix, table format, and chart-axis label."""
        return self.currency_prefix, self.price_format, self.price_axis_label


# The catalogue focuses on widely used exchanges and geographical areas.
# Symbols use Yahoo Finance notation because yfinance is the app's data source.
MANUAL_MARKET_PRESETS: tuple[ManualMarketPreset, ...] = (
    ManualMarketPreset(
        key="us",
        label="United States - NYSE / Nasdaq",
        exchange="New York Stock Exchange and Nasdaq",
        yahoo_suffix="",
        currency_prefix="$",
        price_format="$%.2f",
        price_axis_label="Price (USD)",
        description="Popular U.S. large-cap examples; U.S. symbols normally need no exchange suffix.",
        ticker_examples=(
            ("AAPL", "Apple"),
            ("MSFT", "Microsoft"),
            ("NVDA", "NVIDIA"),
            ("AMZN", "Amazon"),
            ("GOOGL", "Alphabet"),
            ("META", "Meta Platforms"),
            ("JPM", "JPMorgan Chase"),
            ("XOM", "Exxon Mobil"),
        ),
    ),
    ManualMarketPreset(
        key="ireland",
        label="Ireland - Euronext Dublin",
        exchange="Euronext Dublin",
        yahoo_suffix=".IR",
        currency_prefix="\u20ac",
        price_format="\u20ac%.2f",
        price_axis_label="Price (EUR)",
        description="Examples from the app's tracked Irish equity universe.",
        ticker_examples=(
            ("A5G.IR", "AIB Group"),
            ("BIRG.IR", "Bank of Ireland Group"),
            ("KRZ.IR", "Kerry Group"),
            ("KRX.IR", "Kingspan Group"),
            ("RYA.IR", "Ryanair Holdings"),
            ("GL9.IR", "Glanbia"),
            ("UPR.IR", "Uniphar"),
            ("GVR.IR", "Glenveagh Properties"),
        ),
    ),
    ManualMarketPreset(
        key="italy",
        label="Italy - Borsa Italiana",
        exchange="Borsa Italiana",
        yahoo_suffix=".MI",
        currency_prefix="\u20ac",
        price_format="\u20ac%.2f",
        price_axis_label="Price (EUR)",
        description="Large Italian listings available through Yahoo Finance.",
        ticker_examples=(
            ("ENEL.MI", "Enel"),
            ("ENI.MI", "Eni"),
            ("ISP.MI", "Intesa Sanpaolo"),
            ("UCG.MI", "UniCredit"),
            ("STM.MI", "STMicroelectronics"),
            ("G.MI", "Assicurazioni Generali"),
            ("LDO.MI", "Leonardo"),
            ("RACE.MI", "Ferrari"),
        ),
    ),
    ManualMarketPreset(
        key="uk",
        label="United Kingdom - London Stock Exchange",
        exchange="London Stock Exchange",
        yahoo_suffix=".L",
        currency_prefix="GBp ",
        price_format="GBp %.2f",
        price_axis_label="Price (GBp)",
        description=(
            "Large London-listed examples. Yahoo Finance commonly uses the `.L` "
            "suffix and returns these quotes in British pence."
        ),
        ticker_examples=(
            ("SHEL.L", "Shell"),
            ("AZN.L", "AstraZeneca"),
            ("HSBA.L", "HSBC Holdings"),
            ("ULVR.L", "Unilever"),
            ("BP.L", "BP"),
            ("GSK.L", "GSK"),
            ("RIO.L", "Rio Tinto"),
            ("LSEG.L", "London Stock Exchange Group"),
        ),
    ),
    ManualMarketPreset(
        key="germany",
        label="Germany - Xetra",
        exchange="Xetra",
        yahoo_suffix=".DE",
        currency_prefix="\u20ac",
        price_format="\u20ac%.2f",
        price_axis_label="Price (EUR)",
        description="Large German examples using Yahoo Finance's `.DE` Xetra suffix.",
        ticker_examples=(
            ("SAP.DE", "SAP"),
            ("SIE.DE", "Siemens"),
            ("ALV.DE", "Allianz"),
            ("BMW.DE", "BMW"),
            ("MBG.DE", "Mercedes-Benz Group"),
            ("DTE.DE", "Deutsche Telekom"),
            ("BAS.DE", "BASF"),
            ("ADS.DE", "Adidas"),
        ),
    ),
    ManualMarketPreset(
        key="france",
        label="France - Euronext Paris",
        exchange="Euronext Paris",
        yahoo_suffix=".PA",
        currency_prefix="\u20ac",
        price_format="\u20ac%.2f",
        price_axis_label="Price (EUR)",
        description="Large French examples using Yahoo Finance's `.PA` suffix.",
        ticker_examples=(
            ("MC.PA", "LVMH"),
            ("OR.PA", "L'Oreal"),
            ("TTE.PA", "TotalEnergies"),
            ("AIR.PA", "Airbus"),
            ("SAN.PA", "Sanofi"),
            ("BNP.PA", "BNP Paribas"),
            ("SU.PA", "Schneider Electric"),
            ("CS.PA", "AXA"),
        ),
    ),
    ManualMarketPreset(
        key="netherlands",
        label="Netherlands - Euronext Amsterdam",
        exchange="Euronext Amsterdam",
        yahoo_suffix=".AS",
        currency_prefix="\u20ac",
        price_format="\u20ac%.2f",
        price_axis_label="Price (EUR)",
        description="Large Dutch listings using Yahoo Finance's `.AS` suffix.",
        ticker_examples=(
            ("ASML.AS", "ASML"),
            ("ADYEN.AS", "Adyen"),
            ("PHIA.AS", "Philips"),
            ("INGA.AS", "ING Group"),
            ("HEIA.AS", "Heineken"),
            ("KPN.AS", "KPN"),
            ("PRX.AS", "Prosus"),
            ("AD.AS", "Ahold Delhaize"),
        ),
    ),
    ManualMarketPreset(
        key="spain",
        label="Spain - Bolsa de Madrid",
        exchange="Bolsa de Madrid",
        yahoo_suffix=".MC",
        currency_prefix="\u20ac",
        price_format="\u20ac%.2f",
        price_axis_label="Price (EUR)",
        description="Large Spanish examples using Yahoo Finance's `.MC` suffix.",
        ticker_examples=(
            ("SAN.MC", "Banco Santander"),
            ("ITX.MC", "Inditex"),
            ("IBE.MC", "Iberdrola"),
            ("BBVA.MC", "BBVA"),
            ("REP.MC", "Repsol"),
            ("TEF.MC", "Telefonica"),
            ("FER.MC", "Ferrovial"),
            ("CABK.MC", "CaixaBank"),
        ),
    ),
    ManualMarketPreset(
        key="canada",
        label="Canada - Toronto Stock Exchange",
        exchange="Toronto Stock Exchange",
        yahoo_suffix=".TO",
        currency_prefix="C$",
        price_format="C$%.2f",
        price_axis_label="Price (CAD)",
        description="Large Canadian examples using Yahoo Finance's `.TO` suffix.",
        ticker_examples=(
            ("RY.TO", "Royal Bank of Canada"),
            ("TD.TO", "Toronto-Dominion Bank"),
            ("SHOP.TO", "Shopify"),
            ("ENB.TO", "Enbridge"),
            ("CNR.TO", "Canadian National Railway"),
            ("CP.TO", "Canadian Pacific Kansas City"),
            ("SU.TO", "Suncor Energy"),
            ("BNS.TO", "Bank of Nova Scotia"),
        ),
    ),
    ManualMarketPreset(
        key="australia",
        label="Australia - Australian Securities Exchange",
        exchange="Australian Securities Exchange",
        yahoo_suffix=".AX",
        currency_prefix="A$",
        price_format="A$%.2f",
        price_axis_label="Price (AUD)",
        description="Large Australian examples using Yahoo Finance's `.AX` suffix.",
        ticker_examples=(
            ("BHP.AX", "BHP Group"),
            ("CBA.AX", "Commonwealth Bank"),
            ("CSL.AX", "CSL"),
            ("NAB.AX", "National Australia Bank"),
            ("WBC.AX", "Westpac"),
            ("ANZ.AX", "ANZ Group"),
            ("WES.AX", "Wesfarmers"),
            ("MQG.AX", "Macquarie Group"),
        ),
    ),
    ManualMarketPreset(
        key="japan",
        label="Japan - Tokyo Stock Exchange",
        exchange="Tokyo Stock Exchange",
        yahoo_suffix=".T",
        currency_prefix="\u00a5",
        price_format="\u00a5%.2f",
        price_axis_label="Price (JPY)",
        description="Large Japanese examples using Yahoo Finance's `.T` suffix.",
        ticker_examples=(
            ("7203.T", "Toyota Motor"),
            ("6758.T", "Sony Group"),
            ("9984.T", "SoftBank Group"),
            ("8306.T", "Mitsubishi UFJ Financial Group"),
            ("6501.T", "Hitachi"),
            ("6861.T", "Keyence"),
            ("9432.T", "Nippon Telegraph and Telephone"),
            ("8035.T", "Tokyo Electron"),
        ),
    ),
    ManualMarketPreset(
        key="hong_kong",
        label="Hong Kong - Hong Kong Stock Exchange",
        exchange="Hong Kong Stock Exchange",
        yahoo_suffix=".HK",
        currency_prefix="HK$",
        price_format="HK$%.2f",
        price_axis_label="Price (HKD)",
        description="Large Hong Kong examples using Yahoo Finance's `.HK` suffix.",
        ticker_examples=(
            ("0700.HK", "Tencent"),
            ("9988.HK", "Alibaba Group"),
            ("0005.HK", "HSBC Holdings"),
            ("1299.HK", "AIA Group"),
            ("3690.HK", "Meituan"),
            ("0941.HK", "China Mobile"),
            ("2318.HK", "Ping An Insurance"),
            ("0388.HK", "Hong Kong Exchanges and Clearing"),
        ),
    ),
    ManualMarketPreset(
        key="switzerland",
        label="Switzerland - SIX Swiss Exchange",
        exchange="SIX Swiss Exchange",
        yahoo_suffix=".SW",
        currency_prefix="CHF ",
        price_format="CHF %.2f",
        price_axis_label="Price (CHF)",
        description="Large Swiss examples using Yahoo Finance's `.SW` suffix.",
        ticker_examples=(
            ("NESN.SW", "Nestle"),
            ("NOVN.SW", "Novartis"),
            ("ROG.SW", "Roche"),
            ("UBSG.SW", "UBS Group"),
            ("ZURN.SW", "Zurich Insurance Group"),
            ("ABBN.SW", "ABB"),
            ("CFR.SW", "Richemont"),
            ("SREN.SW", "Swiss Re"),
        ),
    ),
    ManualMarketPreset(
        key="indices",
        label="Global - major market indices",
        exchange="Major global benchmark indices",
        yahoo_suffix="",
        currency_prefix="",
        price_format="%.2f",
        price_axis_label="Index level",
        description="Benchmark index symbols from several geographical markets.",
        ticker_examples=(
            ("^GSPC", "S&P 500"),
            ("^DJI", "Dow Jones Industrial Average"),
            ("^IXIC", "Nasdaq Composite"),
            ("^FTSE", "FTSE 100"),
            ("^GDAXI", "DAX"),
            ("^FCHI", "CAC 40"),
            ("FTSEMIB.MI", "FTSE MIB"),
            ("^N225", "Nikkei 225"),
            ("^HSI", "Hang Seng Index"),
        ),
    ),
)

DEFAULT_MANUAL_MARKET = MANUAL_MARKET_PRESETS[0].label
_MANUAL_MARKET_BY_LABEL = {preset.label: preset for preset in MANUAL_MARKET_PRESETS}


def manual_market_labels() -> tuple[str, ...]:
    """Return the sidebar market choices in their intended display order."""
    labels = tuple(preset.label for preset in MANUAL_MARKET_PRESETS)
    bac_log_list_preview("ticker_catalog.manual_market_labels", "labels", list(labels))
    return labels


def get_manual_market_preset(label: str | None) -> ManualMarketPreset:
    """Resolve a market label and safely fall back to the default preset."""
    preset = _MANUAL_MARKET_BY_LABEL.get(label or "", MANUAL_MARKET_PRESETS[0])
    bac_log_kv(
        "ticker_catalog.get_manual_market_preset",
        requested_label=label,
        resolved_label=preset.label,
        yahoo_suffix=preset.yahoo_suffix,
    )
    return preset


def initialize_manual_market_state(session_state: Any) -> None:
    """Keep a stale Streamlit market value from breaking the selectbox."""
    if session_state is None or not hasattr(session_state, "get"):
        return

    try:
        if session_state.get("manual_market") not in _MANUAL_MARKET_BY_LABEL:
            session_state["manual_market"] = DEFAULT_MANUAL_MARKET
            bac_log_section(
                "ticker_catalog.initialize_manual_market_state",
                "Manual market state reset to the default.",
            )
    except Exception as ex:
        # This mirrors the defensive session-state behavior in `app_config.py`.
        bac_log_kv(
            "ticker_catalog.initialize_manual_market_state",
            session_state_error=str(ex),
        )


def format_manual_ticker_option(ticker: str, preset: ManualMarketPreset) -> str:
    """Show a friendly company name while retaining the ticker as the value."""
    company = preset.company_name(ticker)
    formatted = f"{company} ({ticker})" if company != ticker.upper() else ticker.upper()
    bac_log_kv(
        "ticker_catalog.format_manual_ticker_option",
        ticker=ticker,
        company=company,
        formatted=formatted,
    )
    return formatted


def normalize_manual_tickers(
    selected_values: Iterable[str],
    preset: ManualMarketPreset,
    max_tickers: int,
) -> list[str]:
    """Normalize selected or typed symbols for the chosen geographical market.

    Custom entries can be typed individually or comma-separated. When a user
    enters an unqualified symbol for a non-U.S. exchange, the selected market's
    Yahoo suffix is appended automatically. Fully qualified tickers, indices,
    currencies, and other special Yahoo symbols are preserved.
    """
    selected_list = [str(value) for value in selected_values]
    bac_log_list_preview(
        "ticker_catalog.normalize_manual_tickers",
        "selected_values",
        selected_list,
    )
    bac_log_kv(
        "ticker_catalog.normalize_manual_tickers",
        market=preset.label,
        suffix=preset.yahoo_suffix,
        max_tickers=max_tickers,
    )

    normalized: list[str] = []
    seen: set[str] = set()

    for selected_value in selected_list:
        # Supporting commas and semicolons makes pasted watchlists more forgiving.
        split_value = selected_value.replace(";", ",")
        for raw_ticker in split_value.split(","):
            ticker = raw_ticker.strip().upper()
            if not ticker:
                continue

            # A dot normally means the user already supplied an exchange suffix.
            # Caret and equals symbols identify Yahoo index and instrument syntax.
            is_already_qualified = (
                "." in ticker
                or ticker.startswith("^")
                or "=" in ticker
                or "/" in ticker
            )
            if preset.yahoo_suffix and not is_already_qualified:
                ticker = f"{ticker}{preset.yahoo_suffix}"

            if ticker in seen:
                continue

            seen.add(ticker)
            normalized.append(ticker)
            if len(normalized) >= max_tickers:
                bac_log_section(
                    "ticker_catalog.normalize_manual_tickers",
                    "Manual ticker limit reached; remaining selections were ignored.",
                )
                bac_log_list_preview(
                    "ticker_catalog.normalize_manual_tickers",
                    "normalized_tickers",
                    normalized,
                )
                return normalized

    bac_log_list_preview(
        "ticker_catalog.normalize_manual_tickers",
        "normalized_tickers",
        normalized,
    )
    return normalized
