#############################
# author: mrbacco04@gmail.com
# date: October 2026
# file: tests/test_app_logging.py
#############################

"""Level-handling tests for the BAC_LOG helpers."""

from __future__ import annotations

import io
import logging
import unittest

import app_logging


class AppLoggingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = io.StringIO()
        handler = app_logging._LOGGER.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        self.handler = handler
        self.original_stream = self.handler.setStream(self.stream)
        self.original_level = app_logging._LOGGER.level
        app_logging._LOGGER.setLevel(logging.INFO)

    def tearDown(self) -> None:
        self.handler.setStream(self.original_stream)
        app_logging._LOGGER.setLevel(self.original_level)

    def test_debug_detail_is_hidden_at_info(self) -> None:
        app_logging.bac_debug_kv("tests.logging", step=1)
        app_logging.bac_log_kv("tests.logging", status="ready")

        output = self.stream.getvalue()
        self.assertNotIn("step=1", output)
        self.assertIn("[BAC_LOG]", output)
        self.assertIn("INFO | tests.logging | status='ready'", output)

    def test_error_fields_escalate_but_metrics_do_not(self) -> None:
        app_logging.bac_debug_kv("tests.logging", error="provider unavailable")
        app_logging.bac_log_kv("tests.logging", absolute_error=0.5)

        lines = self.stream.getvalue().splitlines()
        self.assertEqual(2, len(lines))
        self.assertIn("WARNING | tests.logging | error='provider unavailable'", lines[0])
        self.assertIn("INFO | tests.logging | absolute_error=0.5", lines[1])


if __name__ == "__main__":
    unittest.main()
