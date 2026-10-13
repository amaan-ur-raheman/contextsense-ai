"""Word2Vec embedding backend using Gensim."""

from __future__ import annotations

from pathlib import Path
from typing import Any, override

import numpy as np

from contextsense_ai.config import settings
from contextsense_ai.embeddings.base import (
    EmbeddingBackend,
    EmbeddingModelInfo,
    EmbeddingResult,
)
from contextsense_ai.exceptions import EmbeddingError
from contextsense_ai.preprocessing import tokenize
from contextsense_ai.utils.logging_config import get_logger

logger = get_logger(__name__)

# Module-level cached model to avoid reloading
_word2vec_model: Any = None
_word2vec_available: bool = False


class Word2VecBackend(EmbeddingBackend):
    """Word2Vec embedding backend using Gensim's KeyedVectors.

    Uses mean pooling over available token vectors.
    """

    def __init__(self) -> None:
        self._model_path: Path | None = None
        self._loaded = False

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            name=settings.default_word2vec_model,
            backend="word2vec",
            dimensions=300,
            is_available=Word2VecBackend.is_model_available(),
            description="Google News Word2Vec (300D, ~3.6GB). "
            "Large model, not downloaded automatically.",
            resource_requirements="~3.6GB disk, ~8GB RAM recommended",
        )

    @staticmethod
    def is_model_available() -> bool:
        """Check if the Word2Vec model has been loaded."""
        return _word2vec_available

    def load(self) -> None:
        """Load the Word2Vec model from local path or download location."""
        global _word2vec_model, _word2vec_available

        if _word2vec_available and _word2vec_model is not None:
            self._loaded = True
            return

        model_file = settings.data_dir / "word2vec" / settings.default_word2vec_model

        if not model_file.exists():
            model_file.parent.mkdir(parents=True, exist_ok=True)
            raise EmbeddingError(
                f"Word2Vec model not found at {model_file}. "
                "Download it manually from:\n"
                f"{settings.word2vec_model_url}\n"
                "Then place it in the data/word2vec/ directory. "
                "Alternatively, use the Sentence Transformers backend "
                "which downloads its model automatically."
            )

        try:
            import gensim.models.keyedvectors as kv

            logger.info("Loading Word2Vec model from %s...", model_file)
            _word2vec_model = kv.KeyedVectors.load_word2vec_format(
                str(model_file), binary=True, limit=500000
            )
            _word2vec_available = True
            self._loaded = True
            logger.info(
                "Word2Vec model loaded: %d words, %d dimensions.",
                len(_word2vec_model.key_to_index),
                _word2vec_model.vector_size,
            )
        except Exception as exc:
            _word2vec_available = False
            raise EmbeddingError(f"Failed to load Word2Vec model: {exc}") from exc

    def release(self) -> None:
        """Release the loaded model."""
        global _word2vec_model, _word2vec_available
        _word2vec_model = None
        _word2vec_available = False
        self._loaded = False

    def is_available(self) -> bool:
        """Check availability (model must be loaded)."""
        return _word2vec_available and self._loaded

    def _get_vector(self, token: str) -> np.ndarray | None:
        """Get vector for a single token, or None if OOV."""
        global _word2vec_model
        if _word2vec_model is None:
            return None
        try:
            vector: np.ndarray = _word2vec_model.get_vector(token.lower())
            return vector
        except KeyError:
            return None

    @override
    def encode_text(self, text: str) -> EmbeddingResult:
        """Encode a text string using Word2Vec mean pooling."""
        if not self._loaded:
            raise EmbeddingError("Word2Vec model not loaded. Call load() first.")

        tokens = tokenize(text)
        vectors: list[np.ndarray] = []
        tokens_used: list[str] = []

        for token in tokens:
            vec = self._get_vector(token)
            if vec is not None:
                vectors.append(vec)
                tokens_used.append(token)

        if not vectors:
            return EmbeddingResult(
                vector=np.zeros(300),
                model_name=self.model_info.name,
                is_valid=False,
                error_message="No known tokens found in text (all OOV).",
                tokens_used=[],
            )

        mean_vec = np.mean(vectors, axis=0).astype(np.float32)
        return EmbeddingResult(
            vector=mean_vec,
            model_name=self.model_info.name,
            is_valid=True,
            tokens_used=tokens_used,
            dimensions=300,
        )

    @override
    def encode_tokens(self, tokens: list[str]) -> EmbeddingResult:
        """Encode a token list using Word2Vec mean pooling."""
        if not self._loaded:
            raise EmbeddingError("Word2Vec model not loaded. Call load() first.")

        vectors: list[np.ndarray] = []
        tokens_used: list[str] = []

        for token in tokens:
            vec = self._get_vector(token)
            if vec is not None:
                vectors.append(vec)
                tokens_used.append(token)

        if not vectors:
            return EmbeddingResult(
                vector=np.zeros(300),
                model_name=self.model_info.name,
                is_valid=False,
                error_message="No known tokens in provided list (all OOV).",
                tokens_used=[],
            )

        mean_vec = np.mean(vectors, axis=0).astype(np.float32)
        return EmbeddingResult(
            vector=mean_vec,
            model_name=self.model_info.name,
            is_valid=True,
            tokens_used=tokens_used,
            dimensions=300,
        )
