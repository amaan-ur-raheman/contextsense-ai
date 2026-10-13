"""Logging configuration for ContextSense AI."""

from __future__ import annotations

import logging
import sys
from typing import Final

LOG_FORMAT: Final = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: Final = "%Y-%m-%d %H:%M:%S"


def setup_logging(*, level: int = logging.INFO) -> None:
    """Configure root logger with a console handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
