#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: app_logging.py
#############################

"""Shared terminal logging helpers for the Streamlit app.

This module keeps the terminal-printing behavior in one place so the rest of
the application can stay focused on data, modeling, and UI rendering.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import Any


def bac_log(message: str) -> None:
    """Print a timestamped debug message that is easy to grep in the terminal."""
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[BAC_LOG] {timestamp} | {message}")


def bac_log_section(section: str, message: str) -> None:
    """Add a simple section prefix so related logs are easier to scan together."""
    bac_log(f"{section} | {message}")


def bac_log_kv(section: str, **values: Any) -> None:
    """Log structured key/value state without needing a full logging framework."""
    if not values:
        bac_log_section(section, "No structured values were provided.")
        return

    ordered_pairs = ", ".join(f"{key}={values[key]!r}" for key in sorted(values))
    bac_log_section(section, ordered_pairs)


def bac_log_list_preview(section: str, label: str, values: Iterable[Any], limit: int = 5) -> None:
    """Log the size of a collection plus a short preview of the first few items."""
    collected_values = list(values)
    preview = collected_values[:limit]
    remaining = max(len(collected_values) - limit, 0)
    message = f"{label} count={len(collected_values)}, preview={preview}"
    if remaining:
        message += f", remaining={remaining}"
    bac_log_section(section, message)
