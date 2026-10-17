"""Tests for context extractor module."""

from __future__ import annotations

import pytest

from contextsense_ai.context_extractor import ContextExtractor
from contextsense_ai.exceptions import DisambiguationError
from contextsense_ai.schemas import ContextStrategy


class TestContextExtractor:
    """Tests for ContextExtractor class."""

    def test_full_sentence_context(self) -> None:
        """Test full sentence context extraction."""
        extractor = ContextExtractor(context_window=5)
        sentence = "The fisherman sat near the bank of the river."
        context = extractor.extract(sentence, "bank", 0, strategy=ContextStrategy.FULL_SENTENCE)
        assert len(context) > 0
        assert "bank" not in context

    def test_local_window_context(self) -> None:
        """Test local window context extraction."""
        extractor = ContextExtractor(context_window=3)
        sentence = "The fisherman sat near the bank of the river."
        context = extractor.extract(sentence, "bank", 0, strategy=ContextStrategy.LOCAL_WINDOW)
        assert len(context) > 0
        assert "bank" not in context

    def test_context_excludes_target(self) -> None:
        """Test that context excludes the target word occurrence."""
        extractor = ContextExtractor(context_window=5)
        sentence = "The cat sat on the mat"
        context = extractor.extract(sentence, "cat", 0, strategy=ContextStrategy.FULL_SENTENCE)
        assert "cat" not in context

    def test_context_with_repeated_word(self) -> None:
        """Test context when target word appears multiple times."""
        extractor = ContextExtractor(context_window=5)
        sentence = "The bank near the bank"
        context = extractor.extract(sentence, "bank", 0, strategy=ContextStrategy.FULL_SENTENCE)
        # The target occurrence is excluded, but other occurrences remain
        assert "bank" in context

    def test_context_window_size(self) -> None:
        """Test that context window size affects extracted context."""
        sentence = "a b c d e f g h i j k l m n o p q r s t u v w x y z"

        extractor_small = ContextExtractor(context_window=2)
        context_small = extractor_small.extract(
            sentence, "m", 0, strategy=ContextStrategy.LOCAL_WINDOW
        )

        extractor_large = ContextExtractor(context_window=10)
        context_large = extractor_large.extract(
            sentence, "m", 0, strategy=ContextStrategy.LOCAL_WINDOW
        )

        assert len(context_large) >= len(context_small)

    def test_invalid_occurrence_index(self) -> None:
        """Test error when occurrence index is out of range."""
        extractor = ContextExtractor(context_window=5)
        with pytest.raises(DisambiguationError):
            extractor.extract("The cat sat", "dog", 0, strategy=ContextStrategy.FULL_SENTENCE)

    def test_empty_context_after_filtering(self) -> None:
        """Test behavior when context becomes empty after filtering."""
        extractor = ContextExtractor(context_window=1)
        sentence = "a bank the"
        context = extractor.extract(sentence, "bank", 0, strategy=ContextStrategy.LOCAL_WINDOW)
        # After filtering stopwords and punctuation, context may be minimal
        assert isinstance(context, list)


class TestContextMetadata:
    """Tests for context extraction with metadata."""

    def test_extract_with_metadata(self) -> None:
        """Test extraction that returns metadata."""
        extractor = ContextExtractor(context_window=5)
        result = extractor.extract_with_metadata(
            "The fisherman sat near the bank of the river.",
            "bank",
            0,
            strategy=ContextStrategy.LOCAL_WINDOW,
        )
        assert "context_tokens" in result
        assert "all_tokens" in result
        assert "target_position" in result
        assert "occurrences" in result
        assert "highlighted_sentence" in result
        assert "strategy" in result

    def test_metadata_includes_highlighted_sentence(self) -> None:
        """Test that metadata includes highlighted sentence."""
        extractor = ContextExtractor(context_window=5)
        result = extractor.extract_with_metadata(
            "The cat sat on the mat",
            "cat",
            0,
        )
        highlighted = result["highlighted_sentence"]
        assert isinstance(highlighted, str)
        assert "**cat**" in highlighted
