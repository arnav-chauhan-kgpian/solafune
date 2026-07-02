"""Structured logging setup.

Provides a single `get_logger` factory that returns a configured logger. All
handlers write to stderr with a compact, timestamped format. If the standard
`RichHandler` is available it is used for prettier output; otherwise a plain
`StreamHandler` is installed.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_INSTALLED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _install_root_handler(level: int) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    root = logging.getLogger()
    root.setLevel(level)
    # remove any pre-existing handlers (Kaggle sometimes pre-configures logging)
    for h in list(root.handlers):
        root.removeHandler(h)
    try:
        from rich.logging import RichHandler  # type: ignore

        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    except Exception:  # pragma: no cover - rich is optional
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.setLevel(level)
    root.addHandler(handler)
    _INSTALLED = True


def get_logger(name: Optional[str] = None, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name. Defaults to the caller's module.
        level: Log level. Defaults to `SOLAFUNE_LOG_LEVEL` env var or INFO.
    """
    if level is None:
        env = os.environ.get("SOLAFUNE_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env, logging.INFO)
    _install_root_handler(level)
    logger = logging.getLogger(name if name else "solafune")
    logger.setLevel(level)
    return logger
