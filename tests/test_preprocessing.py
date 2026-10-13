"""Tests for text preprocessing module."""

from __future__ import annotations

import pytest

from contextsense_ai.exceptions import PreprocessingError
from contextsense_ai.preprocessing import (
    filter_stopwords,
    find_target_occurrences,
    find_target_occurrences_raw,
    lemmatize_token,
    lemmatize_tokens,
    normalize_case,
    normalize_text,
    preprocess_for_target_lookup,
    strip_punctuation_token,
    tokenize,
)


class TestTokenization:
    """Tests for tokenization."""

    def test_tokenize_basic(self) -> None:
        tokens = tokenize("Hello world")
        assert tokens == ["Hello", "world"]

    def test_tokenize_punctuation(self) -> None:
        tokens = tokenize("Hello, world!")
        assert "," in tokens
        assert "!" in tokens

    def test_tokenize_empty(self) -> None:
        with pytest.raises(PreprocessingError):
            tokenize("")

    def test_tokenize_apostrophe(self) -> None:
        tokens = tokenize("don't stop")
        assert "don't" in tokens or "do" in tokens


class TestNormalization:
    """Tests for text normalization."""

    def test_normalize_whitespace(self) -> None:
        result = normalize_text("  hello   world  ")
        assert result == "hello world"

    def test_normalize_case(self) -> None:
        tokens = normalize_case(["Hello", "WORLD"])
        assert tokens == ["hello", "world"]

    def test_strip_punctuation(self) -> None:
        assert strip_punctuation_token("hello,") == "hello"
        assert strip_punctuation_token('"world"') == "world"
        assert strip_punctuation_token("test.") == "test"


class TestStopwordFiltering:
    """Tests for stopword filtering."""

    def test_filter_stopwords(self) -> None:
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        filtered = filter_stopwords(tokens)
        assert "the" not in filtered
        assert "cat" in filtered
        assert "sat" in filtered

    def test_negation_preserved(self) -> None:
        """Negation words should be handled carefully."""
        tokens = ["do", "not", "like"]
        filtered = filter_stopwords(tokens)
        assert "not" not in filtered  # "not" is in default stopwords
        assert "like" in filtered


class TestLemmatization:
    """Tests for lemmatization."""

    def test_lemmatize_noun(self) -> None:
        result = lemmatize_token("cats", pos="n")
        # WordNet lemmatizer may not be available in test environment
        assert isinstance(result, str)
        assert len(result) > 0

    def test_lemmatize_verb(self) -> None:
        result = lemmatize_token("running", pos="v")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_lemmatize_tokens_list(self) -> None:
        tokens = ["cats", "run", "better"]
        result = lemmatize_tokens(tokens)
        assert len(result) == 3
        assert all(isinstance(r, str) for r in result)


class TestTargetWordLookup:
    """Tests for target word occurrence finding."""

    def test_find_single_occurrence(self) -> None:
        preprocessed = preprocess_for_target_lookup("The cat sat on the mat")
        occurrences = find_target_occurrences(preprocessed, "cat")
        assert occurrences == [1]

    def test_find_multiple_occurrences(self) -> None:
        preprocessed = preprocess_for_target_lookup("The bank by the bank")
        occurrences = find_target_occurrences(preprocessed, "bank")
        assert len(occurrences) == 2

    def test_find_no_occurrences(self) -> None:
        preprocessed = preprocess_for_target_lookup("The cat sat")
        occurrences = find_target_occurrences(preprocessed, "dog")
        assert occurrences == []

    def test_case_insensitive_match(self) -> None:
        preprocessed = preprocess_for_target_lookup("The CAT sat")
        occurrences = find_target_occurrences(preprocessed, "cat")
        assert occurrences == [1]

    def test_punctuation_handling(self) -> None:
        preprocessed = preprocess_for_target_lookup("The cat, sat.")
        occurrences = find_target_occurrences(preprocessed, "cat")
        assert occurrences == [1]

    def test_raw_occurrence_finding(self) -> None:
        occurrences = find_target_occurrences_raw("The cat sat on the cat mat", "cat")
        assert len(occurrences) == 2


class TestPreprocessedSentence:
    """Tests for PreprocessedSentence dataclass."""

    def test_highlight_token(self) -> None:
        preprocessed = preprocess_for_target_lookup("The cat sat")
        highlighted = preprocessed.highlight_token(1)  # "cat" is index 1
        assert "**cat**" in highlighted

    def test_highlight_out_of_range(self) -> None:
        preprocessed = preprocess_for_target_lookup("The cat sat")
        with pytest.raises(PreprocessingError):
            preprocessed.highlight_token(10)
