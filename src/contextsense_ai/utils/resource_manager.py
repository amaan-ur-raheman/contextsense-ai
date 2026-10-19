"""Resource management for NLTK data."""

from __future__ import annotations

from pathlib import Path

import nltk

from contextsense_ai.config import settings
from contextsense_ai.utils.logging_config import get_logger

logger = get_logger(__name__)

# The wordnet corpus ships the SemCor sense-frequency counts, so the
# most-frequent-sense baseline needs nothing beyond it. The separate wordnet30
# package is a legacy mirror that still fails to download on current NLTK
# indexes; requesting it only produced spurious errors.
#
# punkt_tab and the perceptron tagger back POS-aware candidate filtering. The
# tagger is optional: without it the system still runs, it just cannot narrow
# candidates by the target word's part of speech.
REQUIRED_PACKAGES = ("wordnet", "omw-1.4", "punkt_tab")
OPTIONAL_PACKAGES = ("averaged_perceptron_tagger_eng",)

_DOWNLOADED = False
_LAST_REPORT: dict[str, str] = {}


def _register_data_paths(data_dir: Path) -> None:
    """Make ``data_dir`` discoverable by NLTK.

    NLTK's default search path contains several directories that do not exist on
    a given machine, which makes ``nltk.data.find`` raise on the first entry it
    cannot stat. Only real directories are kept.
    """
    paths = [str(data_dir), str(Path.home() / ".nltk_data")]
    nltk.data.path = paths + [p for p in nltk.data.path if p not in paths and Path(p).exists()]


def ensure_nltk_resources(*, force: bool = False) -> bool:
    """Ensure required NLTK resources are available.

    Returns True if every required package is usable. Re-attempting is skipped
    once the resources are in place unless ``force`` is set.
    """
    global _DOWNLOADED, _LAST_REPORT

    if _DOWNLOADED and not force:
        return True

    data_dir = settings.nltk_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    _register_data_paths(data_dir)

    report: dict[str, str] = {}
    ok = True

    for package in (*REQUIRED_PACKAGES, *OPTIONAL_PACKAGES):
        optional = package in OPTIONAL_PACKAGES
        if _ensure_package(package, data_dir):
            report[package] = "available"
        elif optional:
            report[package] = "unavailable (optional)"
            logger.warning("Optional NLTK resource '%s' is unavailable.", package)
        else:
            report[package] = "FAILED"
            ok = False

    _LAST_REPORT = report

    if ok:
        try:
            nltk.corpus.wordnet.synsets("test")
            _DOWNLOADED = True
            logger.info("NLTK resources verified successfully.")
        except Exception as exc:
            logger.error("NLTK resource verification failed: %s", exc)
            _DOWNLOADED = False
            ok = False

    return ok


def _ensure_package(package: str, data_dir: Path) -> bool:
    """Download ``package`` if it is not already present."""
    try:
        nltk.data.find(f"corpora/{package}")
        return True
    except LookupError:
        pass

    logger.info("Downloading NLTK resource: %s", package)
    try:
        if nltk.download(package, download_dir=str(data_dir), quiet=True):
            return True
    except Exception as exc:
        logger.error("Failed to download NLTK resource '%s': %s", package, exc)
    return False


def resource_report() -> dict[str, str]:
    """Return the status of each resource checked by the last call."""
    return dict(_LAST_REPORT)
