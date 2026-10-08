#############################
# author: mrbacco04@gmail.com
# date: July 2026
# file: app_logging.py
#############################

"""Shared terminal logging helpers for the Streamlit app and workers.

The helpers are thin wrappers around the standard `logging` module, so the
verbosity is controlled by the ``LOG_LEVEL`` environment variable (default
``INFO``) while every line keeps its greppable ``[BAC_LOG]`` prefix:

- `bac_log_*` helpers log at INFO.
- `bac_debug_*` helpers log at DEBUG; use them for per-step or per-call detail.
- Any structured log that carries an error field (``error``, ``error_type``,
  or a ``*_error`` key other than a metric such as ``absolute_error``) is
  raised to WARNING automatically, whichever helper emitted it.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable
from typing import Any

LOGGER_NAME = "bac"


def _configure_logger() -> logging.Logger:
    """Attach one stdout handler with the historical BAC_LOG line format."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[BAC_LOG] %(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        # Streamlit configures the root logger; propagating would duplicate lines.
        logger.propagate = False
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    return logger


_LOGGER = _configure_logger()


def _is_error_field(key: str) -> bool:
    return key in {"error", "error_type"} or (
        key.endswith("_error") and not key.endswith("absolute_error")
    )


def _emit_kv(level: int, section: str, values: dict[str, Any]) -> None:
    if any(_is_error_field(key) for key in values):
        level = max(level, logging.WARNING)
    # Formatting every value is the expensive part; skip it when filtered out.
    if not _LOGGER.isEnabledFor(level):
        return
    if not values:
        _LOGGER.log(level, "%s | No structured values were provided.", section)
        return
    ordered_pairs = ", ".join(f"{key}={values[key]!r}" for key in sorted(values))
    _LOGGER.log(level, "%s | %s", section, ordered_pairs)


def _emit_list_preview(
    level: int,
    section: str,
    label: str,
    values: Iterable[Any],
    limit: int,
) -> None:
    if not _LOGGER.isEnabledFor(level):
        return
    collected_values = list(values)
    preview = collected_values[:limit]
    remaining = max(len(collected_values) - limit, 0)
    message = f"{label} count={len(collected_values)}, preview={preview}"
    if remaining:
        message += f", remaining={remaining}"
    _LOGGER.log(level, "%s | %s", section, message)


def bac_log(message: str) -> None:
    """Log one timestamped INFO message that is easy to grep in the terminal."""
    _LOGGER.info("%s", message)


def bac_log_section(section: str, message: str) -> None:
    """Add a simple section prefix so related logs are easier to scan together."""
    _LOGGER.info("%s | %s", section, message)


def bac_debug_section(section: str, message: str) -> None:
    """DEBUG-level variant of `bac_log_section`."""
    _LOGGER.debug("%s | %s", section, message)


def bac_log_kv(section: str, **values: Any) -> None:
    """Log structured key/value state without needing a full logging framework."""
    _emit_kv(logging.INFO, section, values)


def bac_debug_kv(section: str, **values: Any) -> None:
    """DEBUG-level variant of `bac_log_kv` for per-step and per-call detail."""
    _emit_kv(logging.DEBUG, section, values)


def bac_log_list_preview(section: str, label: str, values: Iterable[Any], limit: int = 5) -> None:
    """Log the size of a collection plus a short preview of the first few items."""
    _emit_list_preview(logging.INFO, section, label, values, limit)


def bac_debug_list_preview(
    section: str,
    label: str,
    values: Iterable[Any],
    limit: int = 5,
) -> None:
    """DEBUG-level variant of `bac_log_list_preview`."""
    _emit_list_preview(logging.DEBUG, section, label, values, limit)
