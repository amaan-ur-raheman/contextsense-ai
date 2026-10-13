"""Cosine similarity and vector normalization utilities."""

from __future__ import annotations

import math

import numpy as np

from contextsense_ai.exceptions import EmbeddingError


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.

    Returns a zero vector if the input has zero norm.
    """
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0.0 or not math.isfinite(norm):
        return np.zeros_like(vector)
    normalized: np.ndarray = vector / norm
    return normalized


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
    *,
    normalize: bool = True,
) -> float:
    """Calculate cosine similarity between two vectors.

    cosine_similarity(A, B) = dot(A, B) / (norm(A) × norm(B))

    Args:
        a: First vector.
        b: Second vector.
        normalize: If True, normalize vectors before computing similarity.

    Returns:
        Cosine similarity score in [-1, 1].

    Raises:
        EmbeddingError: If vectors are invalid (empty, mismatched, non-finite, zero norm).
    """
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()

    if a.size == 0 or b.size == 0:
        raise EmbeddingError("Cannot compute similarity: one or both vectors are empty.")

    if a.shape != b.shape:
        raise EmbeddingError(
            f"Dimension mismatch: vector A has {a.shape[0]} dims, vector B has {b.shape[0]} dims."
        )

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise EmbeddingError("Cannot compute similarity: input contains non-finite values.")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        raise EmbeddingError("Cannot compute similarity: one or both vectors have zero norm.")

    if not math.isfinite(norm_a) or not math.isfinite(norm_b):
        raise EmbeddingError("Cannot compute similarity: norm is not finite.")

    if normalize:
        a = a / norm_a
        b = b / norm_b

    similarity = float(np.dot(a, b))

    if not math.isfinite(similarity):
        raise EmbeddingError("Similarity computation produced non-finite result.")

    return max(-1.0, min(1.0, similarity))


def cosine_similarity_safe(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[bool, float | None, str | None]:
    """Safe cosine similarity that returns (success, score, error) tuple.

    Useful when similarity calculation failures should be handled gracefully
    rather than raising exceptions.
    """
    try:
        score = cosine_similarity(a, b)
        return (True, score, None)
    except EmbeddingError as exc:
        return (False, None, str(exc))
