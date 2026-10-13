"""Resource management for NLTK data and embedding model caching."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import nltk
from nltk.corpus.reader.wordnet import WordNetCorpusReader

from contextsense_ai.config import settings
from contextsense_ai.exceptions import ResourceError
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


def get_wordnet_reader() -> WordNetCorpusReader:
    """Return the WordNet corpus reader, downloading if necessary."""
    if not ensure_nltk_resources():
        raise ResourceError(
            "WordNet resources not available. Run: "
            "uv run python -m contextsense_ai.scripts.download_resources"
        )
    return nltk.corpus.wordnet


class ResourceManager:
    """Manages the lifecycle of heavy resources (embedding models).

    Loaders are registered up front and invoked lazily on first access, so an
    expensive model is only loaded when something actually needs it.
    """

    def __init__(self) -> None:
        self._loaders: dict[str, Callable[[], object]] = {}
        self._models: dict[str, object] = {}
        self._status: dict[str, str] = {}

    def register(
        self,
        name: str,
        loader: Callable[[], object],
        *,
        description: str = "",
    ) -> None:
        """Register a model loader without immediately loading it."""
        if description:
            logger.debug("Registered resource '%s': %s", name, description)
        self._loaders[name] = loader
        self._models.pop(name, None)
        self._status[name] = "registered"

    def get(self, name: str) -> object:
        """Get a model, loading it on first access."""
        if name in self._models:
            return self._models[name]

        loader = self._loaders.get(name)
        if loader is None:
            raise ResourceError(f"Unknown resource: '{name}'.")

        try:
            model = loader()
        except Exception:
            self._status[name] = "error"
            logger.exception("Failed to load resource '%s'", name)
            raise

        self._models[name] = model
        self._status[name] = "loaded"
        logger.info("Loaded resource: %s", name)
        return model

    def status(self, name: str) -> str:
        """Return the current status of a named resource."""
        return self._status.get(name, "unknown")

    def release(self, name: str) -> None:
        """Release a single loaded resource."""
        if self._models.pop(name, None) is not None:
            logger.info("Released resource: %s", name)

    def release_all(self) -> None:
        """Release every loaded resource."""
        for name in list(self._models):
            self.release(name)
