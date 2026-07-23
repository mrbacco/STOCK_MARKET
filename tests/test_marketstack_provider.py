#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: tests/test_marketstack_provider.py
#############################

"""Offline contract tests for the optional Marketstack EOD adapter."""

from __future__ import annotations

import unittest

import pandas as pd

from marketstack_provider import fetch_marketstack_history, load_symbol_map


class _Response:
    """Tiny requests-compatible response double; no network is used in tests."""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "data": [
                {
                    "date": "2026-07-22T00:00:00+0000",
                    "open": 201.0,
                    "high": 205.0,
                    "low": 200.0,
                    "close": 204.0,
                    "volume": 1200000,
                },
                {
                    "date": "2026-07-23T00:00:00+0000",
                    "open": 204.0,
                    "high": 207.0,
                    "low": 203.0,
                    "close": 206.0,
                    "volume": 1300000,
                },
            ]
        }


class _Session:
    def __init__(self) -> None:
        self.last_params: dict[str, object] = {}

    def get(self, _endpoint: str, *, params: dict[str, object], timeout: int) -> _Response:
        self.last_params = params
        self.timeout = timeout
        return _Response()


class MarketstackProviderTest(unittest.TestCase):
    def test_eod_payload_is_normalized_for_forecasting(self):
        """Provider field names should not leak into forecasting code."""
        session = _Session()
        history = fetch_marketstack_history(
            "AAPL",
            "1y",
            "1d",
            api_key="test-key",
            session=session,
        )

        self.assertEqual(
            ["Date", "Open", "High", "Low", "Close", "Volume"],
            list(history.columns),
        )
        self.assertEqual(2, len(history))
        self.assertEqual(206.0, history.iloc[-1]["Close"])
        self.assertEqual("AAPL", session.last_params["symbols"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(history["Date"]))

    def test_intraday_is_rejected_instead_of_mislabeled(self):
        """An EOD plan must never be presented as a real-time data source."""
        with self.assertRaisesRegex(ValueError, "interval='1d'"):
            fetch_marketstack_history(
                "AAPL",
                "1d",
                "1m",
                api_key="test-key",
                session=_Session(),
            )

    def test_symbol_map_requires_a_json_object(self):
        self.assertEqual(
            {"ENI.MI": "XMIL:ENI"},
            load_symbol_map('{"eni.mi": "XMIL:ENI"}'),
        )
        with self.assertRaisesRegex(ValueError, "JSON object"):
            load_symbol_map('["AAPL"]')


if __name__ == "__main__":
    unittest.main()
