"""Sentence Transformers embedding backend."""

from __future__ import annotations

from typing import override

import numpy as np
from sentence_transformers import SentenceTransformer

from contextsense_ai.config import settings
from contextsense_ai.embeddings.base import (
    EmbeddingBackend,
    EmbeddingModelInfo,
    EmbeddingResult,
)
from contextsense_ai.exceptions import EmbeddingError
from contextsense_ai.utils.logging_config import get_logger

logger = get_logger(__name__)

# Module-level cache
_st_model: SentenceTransformer | None = None
_st_available: bool = False


class SentenceTransformerBackend(EmbeddingBackend):
    """Sentence Transformers backend for contextual embeddings.

    Uses a lightweight pretrained model for encoding context and sense descriptions.
    """

    def __init__(self) -> None:
        self._loaded = False
        self._model_name = settings.default_sentence_transformer_model

    @property
    def model_info(self) -> EmbeddingModelInfo:
        """Return model metadata. Dimensions are approximate."""
        dims = self._get_model_dimensions(self._model_name)
        return EmbeddingModelInfo(
            name=self._model_name,
            backend="sentence_transformers",
            dimensions=dims,
            is_available=SentenceTransformerBackend.is_model_available(),
            description=f"Sentence Transformers model '{self._model_name}'. "
            "Downloads automatically on first use (~80-200MB).",
            resource_requirements="~100-300MB disk, ~500MB-2GB RAM",
        )

    @staticmethod
    def _get_model_dimensions(model_name: str) -> int:
        """Try to determine embedding dimensions without loading."""
        known_dims = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-MiniLM-L3-v2": 256,
            "all-distilroberta-v1": 768,
        }
        return known_dims.get(model_name, 384)

    @staticmethod
    def is_model_available() -> bool:
        """Check if the Sentence Transformer model is loaded."""
        return _st_available

    def load(self) -> None:
        """Load the Sentence Transformer model."""
        global _st_model, _st_available

        if _st_available and _st_model is not None:
            self._loaded = True
            return

        try:
            logger.info("Loading Sentence Transformer model: %s", self._model_name)
            _st_model = SentenceTransformer(self._model_name)
            _st_available = True
            self._loaded = True
            logger.info("Sentence Transformer model loaded successfully.")
        except Exception as exc:
            _st_available = False
            raise EmbeddingError(
                f"Failed to load Sentence Transformer model '{self._model_name}': {exc}. "
                "Check network connectivity and disk space."
            ) from exc

    def release(self) -> None:
        """Release the model from memory."""
        global _st_model, _st_available
        if _st_model is not None:
            del _st_model
        _st_model = None
        _st_available = False
        self._loaded = False
        logger.info("Sentence Transformer model released.")

    def is_available(self) -> bool:
        """Check availability."""
        return _st_available and self._loaded

    @override
    def encode_text(self, text: str) -> EmbeddingResult:
        """Encode text using Sentence Transformers."""
        if not self._loaded:
            raise EmbeddingError("Sentence Transformer model not loaded. Call load() first.")

        if not text or not text.strip():
            return EmbeddingResult(
                vector=np.zeros(self.model_info.dimensions or 384),
                model_name=self.model_info.name,
                is_valid=False,
                error_message="Empty text provided for encoding.",
            )

        try:
            global _st_model
            if _st_model is None:
                raise EmbeddingError("Sentence Transformer model was released; call load() first.")
            embedding = _st_model.encode(text, normalize_embeddings=False).astype(np.float32)
            embedding = np.array(embedding).flatten()
            return EmbeddingResult(
                vector=embedding,
                model_name=self.model_info.name,
                is_valid=True,
                dimensions=int(embedding.shape[-1])
                if embedding.ndim > 0
                else self.model_info.dimensions or 384,
            )
        except Exception as exc:
            raise EmbeddingError(f"Sentence Transformer encoding failed: {exc}") from exc

    @override
    def encode_tokens(self, tokens: list[str]) -> EmbeddingResult:
        """Encode tokens by joining them into a text string.

        Sentence Transformers expect full text, not token lists.
        This method joins tokens and encodes the resulting text.
        """
        if not tokens:
            return EmbeddingResult(
                vector=np.zeros(self.model_info.dimensions or 384),
                model_name=self.model_info.name,
                is_valid=False,
                error_message="Empty token list provided.",
            )

        text = " ".join(str(t) for t in tokens)
        return self.encode_text(text)
