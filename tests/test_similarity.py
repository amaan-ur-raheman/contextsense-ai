"""Tests for cosine similarity module."""

from __future__ import annotations

import numpy as np
import pytest

from contextsense_ai.disambiguation.similarity import (
    cosine_similarity,
    cosine_similarity_safe,
    normalize_vector,
)
from contextsense_ai.exceptions import EmbeddingError


class TestNormalizeVector:
    """Tests for vector normalization."""

    def test_normalize_unit_vector(self) -> None:
        """Test normalizing an already unit vector."""
        v = np.array([1.0, 0.0, 0.0])
        result = normalize_vector(v)
        assert np.allclose(result, v)

    def test_normalize_zero_vector(self) -> None:
        """Test normalizing a zero vector returns zero."""
        v = np.array([0.0, 0.0, 0.0])
        result = normalize_vector(v)
        assert np.allclose(result, 0.0)

    def test_normalize_known_vector(self) -> None:
        """Test normalizing a vector with known norm."""
        v = np.array([3.0, 4.0])
        result = normalize_vector(v)
        assert np.allclose(np.linalg.norm(result), 1.0)

    def test_normalize_preserves_direction(self) -> None:
        """Test that normalization preserves direction."""
        v = np.array([1.0, 2.0, 3.0])
        normalized = normalize_vector(v)
        # Cross product should be near zero (parallel vectors)
        cross = np.cross(v, normalized)
        assert np.allclose(cross, 0.0, atol=1e-10)


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self) -> None:
        """Test cosine similarity of identical vectors."""
        v = np.array([1.0, 0.0, 0.0])
        similarity = cosine_similarity(v, v)
        assert np.isclose(similarity, 1.0)

    def test_orthogonal_vectors(self) -> None:
        """Test cosine similarity of orthogonal vectors."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        similarity = cosine_similarity(v1, v2)
        assert np.isclose(similarity, 0.0)

    def test_opposite_vectors(self) -> None:
        """Test cosine similarity of opposite vectors."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        similarity = cosine_similarity(v1, v2)
        assert np.isclose(similarity, -1.0)

    def test_known_similarity(self) -> None:
        """Test cosine similarity with known values."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([1.0, 1.0])
        # cos = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2)
        expected = 1.0 / np.sqrt(2.0)
        similarity = cosine_similarity(v1, v2)
        assert np.isclose(similarity, expected, rtol=1e-5)

    def test_empty_vector_raises(self) -> None:
        """Test that empty vectors raise an error."""
        v1 = np.array([])
        v2 = np.array([1.0, 0.0])
        with pytest.raises(EmbeddingError):
            cosine_similarity(v1, v2)

    def test_mismatched_dimensions_raises(self) -> None:
        """Test that mismatched dimensions raise an error."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        with pytest.raises(EmbeddingError):
            cosine_similarity(v1, v2)

    def test_zero_norm_raises(self) -> None:
        """Test that zero norm raises an error."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 0.0])
        with pytest.raises(EmbeddingError):
            cosine_similarity(v1, v2)

    def test_non_finite_values_raises(self) -> None:
        """Test that NaN values raise an error."""
        v1 = np.array([1.0, np.nan])
        v2 = np.array([1.0, 0.0])
        with pytest.raises(EmbeddingError):
            cosine_similarity(v1, v2)

    def test_similarity_in_range(self) -> None:
        """Test that similarity is always in [-1, 1]."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        similarity = cosine_similarity(v1, v2)
        assert -1.0 <= similarity <= 1.0

    def test_positive_similarity_for_similar_vectors(self) -> None:
        """Test that similar vectors have positive similarity."""
        v1 = np.array([1.0, 1.0, 1.0])
        v2 = np.array([1.1, 1.1, 1.1])
        similarity = cosine_similarity(v1, v2)
        assert similarity > 0.9

    def test_clipped_similarity(self) -> None:
        """Test that similarity is clipped to [-1, 1]."""
        # Due to floating point, similarity might slightly exceed bounds
        v1 = np.array([1.0, 0.0])
        v2 = np.array([1.0, 0.0])
        similarity = cosine_similarity(v1, v2)
        assert -1.0 <= similarity <= 1.0


class TestCosineSimilaritySafe:
    """Tests for safe cosine similarity that returns tuples."""

    def test_successful_similarity(self) -> None:
        """Test successful similarity calculation."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        success, score, error = cosine_similarity_safe(v1, v2)
        assert success is True
        assert score == 0.0
        assert error is None

    def test_failed_similarity(self) -> None:
        """Test failed similarity returns error message."""
        v1 = np.array([])
        v2 = np.array([1.0, 0.0])
        success, score, error = cosine_similarity_safe(v1, v2)
        assert success is False
        assert score is None
        assert error is not None
        assert len(error) > 0
