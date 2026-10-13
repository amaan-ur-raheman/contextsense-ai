"""Pytest configuration and fixtures for ContextSense AI tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is on the path for tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _setup_nltk_for_testing() -> None:
    """Ensure NLTK can find data in ~/.nltk_data."""
    import nltk

    actual_data_dir = Path.home() / ".nltk_data"
    if not actual_data_dir.exists():
        return

    # Replace non-existent paths and put actual_data_dir first
    nltk.data.path = [
        str(actual_data_dir),
    ] + [str(p) for p in nltk.data.path if Path(p).exists() and str(p) != str(actual_data_dir)]

    os.environ.setdefault("NLTK_DATA", str(actual_data_dir))


def pytest_configure() -> None:
    """Configure pytest — runs before test collection."""
    _setup_nltk_for_testing()

    # Ensure NLTK resources are available
    try:
        from contextsense_ai.utils.resource_manager import ensure_nltk_resources

        ensure_nltk_resources()
    except Exception:
        pass
