"""Configuration settings for ContextSense AI."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logging.getLogger(__name__).addHandler(logging.NullHandler())


class Settings:
    """Application configuration.

    Settings are loaded from environment variables or a .env file.
    """

    # Data directories
    data_dir: Path = Path(os.environ.get("DATA_DIR", "data"))
    raw_data_dir: Path = Path(os.environ.get("RAW_DATA_DIR", "data/raw"))
    processed_data_dir: Path = Path(os.environ.get("PROCESSED_DATA_DIR", "data/processed"))
    artifacts_dir: Path = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
    cache_dir: Path = Path(os.environ.get("CACHE_DIR", "~/.cache/contextsense_ai")).expanduser()

    # NLTK configuration
    nltk_data_dir: Path = Path(os.environ.get("NLTK_DATA_DIR", "~/.nltk_data")).expanduser()
    nltk_download_timeout: int = int(os.environ.get("NLTK_DOWNLOAD_TIMEOUT", "300"))

    # Default embedding model
    default_embedding_backend: str = os.environ.get(
        "DEFAULT_EMBEDDING_BACKEND", "sentence_transformer"
    )
    default_sentence_transformer_model: str = os.environ.get(
        "DEFAULT_SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
    )
    default_word2vec_model: str = os.environ.get(
        "DEFAULT_WORD2VEC_MODEL", "GoogleNews-vectors-negative300.bin.gz"
    )
    word2vec_model_url: str = (
        "https://storage.googleapis.com/word2vec-tutorial-models/"
        "GoogleNews-vectors-negative300.bin.gz"
    )

    # Context extraction defaults
    default_context_window: int = int(os.environ.get("DEFAULT_CONTEXT_WINDOW", "5"))
    min_context_window: int = int(os.environ.get("MIN_CONTEXT_WINDOW", "1"))
    max_context_window: int = int(os.environ.get("MAX_CONTEXT_WINDOW", "20"))

    # Evaluation defaults
    default_test_split: float = float(os.environ.get("DEFAULT_TEST_SPLIT", "0.2"))
    random_seed: int = int(os.environ.get("RANDOM_SEED", "42"))

    @property
    def nltk_data_path(self) -> list[Path]:
        """Return NLTK data paths ensuring the custom directory exists."""
        path = self.nltk_data_dir
        path.mkdir(parents=True, exist_ok=True)
        return [path]

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for directory in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.artifacts_dir,
            self.cache_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
"""Global settings instance."""
