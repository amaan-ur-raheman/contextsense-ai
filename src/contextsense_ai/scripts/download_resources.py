#!/usr/bin/env python3
"""Download required NLTK resources and prepare the environment."""

from __future__ import annotations

import sys

from contextsense_ai.config import settings
from contextsense_ai.utils.resource_manager import (
    REQUIRED_PACKAGES,
    ensure_nltk_resources,
    resource_report,
)


def main() -> None:
    """Download and verify the NLTK resources the project needs."""
    print("=" * 60)
    print("ContextSense AI - Resource Setup")
    print("=" * 60)
    print()

    settings.ensure_directories()

    print(f"NLTK data directory: {settings.nltk_data_dir}")
    print(f"Data directory:      {settings.data_dir}")
    print()

    print("Checking NLTK resources...")
    if ensure_nltk_resources():
        print()
        print("✓ NLTK resources are available.")
        print()
        print("Resource status:")
        for package, status in resource_report().items():
            print(f"  - {package}: {status}")
        return

    print()
    print("✗ Failed to prepare NLTK resources.")
    print()
    print("Try again with:")
    print("  uv run python -m contextsense_ai.scripts.download_resources")
    print()
    print("Or download manually with:")
    print(f"  python -m nltk.downloader {' '.join(REQUIRED_PACKAGES)}")
    sys.exit(1)


if __name__ == "__main__":
    main()
