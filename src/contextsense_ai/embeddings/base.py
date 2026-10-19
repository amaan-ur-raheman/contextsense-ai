"""Base embedding interface and data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from contextsense_ai.exceptions import EmbeddingError


@dataclass
class EmbeddingModelInfo:
    """Metadata about an embedding model."""

    name: str
    backend: str
    dimensions: int | None = None
    is_available: bool = False
    description: str = ""
    resource_requirements: str = ""

    @property
    def status_text(self) -> str:
        """Human-readable availability status."""
        if self.is_available:
            return f"Available ({self.dimensions or '?'}D)"
        return "Not available"


@dataclass
class EmbeddingResult:
    """Result of encoding text with an embedding model."""

    vector: np.ndarray
    model_name: str
    is_valid: bool
    error_message: str | None = None
    tokens_used: list[str] | None = None
    dimensions: int = 0

    def __post_init__(self) -> None:
        if self.vector.size > 0:
            self.dimensions = int(self.vector.shape[-1])

    @property
    def is_zero_vector(self) -> bool:
        """Check if this is effectively a zero vector."""
        return self.vector.size == 0 or np.allclose(self.vector, 0.0)

    @property
    def has_non_finite_values(self) -> bool:
        """Check for NaN or Inf values."""
        return not np.all(np.isfinite(self.vector))


# Type alias for vector data
Vector = np.ndarray


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @property
    @abstractmethod
    def model_info(self) -> EmbeddingModelInfo:
        """Return metadata about the embedding model."""

    @abstractmethod
    def encode_text(self, text: str) -> EmbeddingResult:
        """Encode a piece of text into a single vector."""

    @abstractmethod
    def encode_tokens(self, tokens: list[str]) -> EmbeddingResult:
        """Encode a list of tokens into a single vector (mean pooling)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the backend and its model are available."""

    @property
    def cache_identity(self) -> str:
        """A key that changes whenever this backend would produce different vectors.

        Callers memoise per-model artifacts under this. It must separate two
        backends that report the same ``model_info.name`` but can hold
        different vectors.
        """
        info = self.model_info
        return info.name if info else "backend"

    @abstractmethod
    def load(self) -> None:
        """Load or initialize the model."""

    @abstractmethod
    def release(self) -> None:
        """Release resources held by the model."""

    def encode_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Encode multiple texts efficiently if supported.

        Default implementation calls encode_text for each item.
        Subclasses should override for batch efficiency.
        """
        return [self.encode_text(text) for text in texts]

    def validate_result(self, result: EmbeddingResult) -> None:
        """Validate an embedding result and raise on problems."""
        if not result.is_valid:
            raise EmbeddingError(f"Embedding failed: {result.error_message or 'Unknown error'}.")
        if result.is_zero_vector:
            raise EmbeddingError("Embedding produced a zero-length vector.")
        if result.has_non_finite_values:
            raise EmbeddingError("Embedding contains non-finite values (NaN/Inf).")

    def ensure_available(self) -> None:
        """Ensure the backend is available, loading if necessary."""
        if not self.is_available():
            self.load()
        if not self.is_available():
            raise EmbeddingError(f"Embedding backend '{self.model_info.name}' is not available.")
