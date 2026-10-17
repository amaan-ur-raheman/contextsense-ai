"""Tests for disambiguation engine module."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from contextsense_ai.disambiguation.engine import build_sense_description, disambiguate
from contextsense_ai.embeddings.base import EmbeddingBackend, EmbeddingModelInfo, EmbeddingResult
from contextsense_ai.schemas import (
    CandidateSense,
    ContextStrategy,
    ModelVariant,
    PredictionResult,
)


def _check_wordnet_available() -> bool:
    """Check if WordNet resources are available."""
    try:
        from contextsense_ai.utils.resource_manager import ensure_nltk_resources

        return ensure_nltk_resources()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Mock embedding backend for testing
# ---------------------------------------------------------------------------


@dataclass
class MockEmbeddingBackend(EmbeddingBackend):
    """Mock embedding backend that returns controlled vectors."""

    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    model_name: str = "mock"
    dimensions: int = 3
    fail_on: list[str] | None = None
    return_zero_vector: bool = False

    @property
    def model_info(self) -> EmbeddingModelInfo:
        """Return mock model info."""
        return EmbeddingModelInfo(
            name=self.model_name,
            backend="mock",
            dimensions=self.dimensions,
            is_available=True,
            description="Mock embedding backend for testing",
        )

    def encode_text(self, text: str) -> EmbeddingResult:
        """Return a mock vector for the text."""
        if self.fail_on and any(fail in text for fail in self.fail_on):
            return EmbeddingResult(
                vector=np.zeros(self.dimensions),
                model_name=self.model_name,
                is_valid=False,
                error_message=f"Mock failure for: {text}",
            )

        if self.return_zero_vector:
            return EmbeddingResult(
                vector=np.zeros(self.dimensions),
                model_name=self.model_name,
                is_valid=False,
                error_message="Mock zero vector",
            )

        # Generate deterministic vector from text hash
        seed = sum(ord(c) for c in text) % 1000
        np.random.seed(seed)
        vec = np.random.randn(self.dimensions).astype(np.float32)

        return EmbeddingResult(
            vector=vec,
            model_name=self.model_name,
            is_valid=True,
            dimensions=self.dimensions,
        )

    def encode_tokens(self, tokens: list[str]) -> EmbeddingResult:
        """Encode tokens by joining and encoding."""
        return self.encode_text(" ".join(str(t) for t in tokens))

    def is_available(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def release(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _check_wordnet_available(), reason="WordNet resources not available")
class TestBuildSenseDescription:
    """Tests for sense description building."""

    def test_build_description_includes_definition(self) -> None:
        """Test that description includes the definition."""
        from nltk.corpus import wordnet

        syn = wordnet.synset("bank.n.01")
        candidate = CandidateSense(
            synset=syn,
            pos="n",
            lemmas=["bank"],
            definition="A financial institution.",
            examples=["I went to the bank."],
            hypernyms=["financial_institution"],
            hyponyms=[],
        )
        desc = build_sense_description(candidate)
        assert "financial institution" in desc.lower()

    def test_build_description_includes_examples(self) -> None:
        """Test that description includes examples."""
        from nltk.corpus import wordnet

        syn = wordnet.synset("bat.n.02")
        candidate = CandidateSense(
            synset=syn,
            pos="n",
            lemmas=["bat"],
            definition="A flying mammal.",
            examples=["A bat flew out of the cave."],
            hypernyms=[],
            hyponyms=[],
        )
        desc = build_sense_description(candidate)
        assert "bat flew" in desc.lower() or "example" in desc.lower()

    def test_build_description_includes_lemmas(self) -> None:
        """Test that description includes lemmas."""
        from nltk.corpus import wordnet

        syn = wordnet.synset("mouse.n.03")
        candidate = CandidateSense(
            synset=syn,
            pos="n",
            lemmas=["mouse", "computer_mouse"],
            definition="A computer pointing device.",
            examples=[],
            hypernyms=[],
            hyponyms=[],
        )
        desc = build_sense_description(candidate)
        assert "mouse" in desc.lower()


@pytest.mark.skipif(not _check_wordnet_available(), reason="WordNet resources not available")
class TestDisambiguateEngine:
    """Tests for the disambiguate function."""

    @pytest.fixture
    def mock_backend(self) -> MockEmbeddingBackend:
        """Create a mock embedding backend."""
        return MockEmbeddingBackend()

    def test_disambiguate_with_real_wordnet(self, mock_backend: MockEmbeddingBackend) -> None:
        """Test disambiguation with real WordNet candidates."""
        result = disambiguate(
            sentence="I deposited money in the bank.",
            target_word="bank",
            occurrence_index=0,
            model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
            embedding_backend=mock_backend,
            context_strategy=ContextStrategy.LOCAL_WINDOW,
        )

        assert isinstance(result, PredictionResult)
        assert result.target_word == "bank"
        assert result.sentence == "I deposited money in the bank."

    def test_first_sense_baseline(self, mock_backend: MockEmbeddingBackend) -> None:
        """Test first-sense baseline method."""
        result = disambiguate(
            sentence="The bank approved the loan.",
            target_word="bank",
            occurrence_index=0,
            model_variant=ModelVariant.FIRST_SENSE,
            embedding_backend=mock_backend,
        )

        assert result.success is True
        assert result.predicted_synset_id  # Should have a prediction
        assert result.model_variant == ModelVariant.FIRST_SENSE

    def test_most_frequent_sense_baseline(self, mock_backend: MockEmbeddingBackend) -> None:
        """Test most-frequent-sense baseline method."""
        result = disambiguate(
            sentence="The bank approved the loan.",
            target_word="bank",
            occurrence_index=0,
            model_variant=ModelVariant.MOST_FREQUENT_SENSE,
            embedding_backend=mock_backend,
        )

        assert result.success is True
        assert result.model_variant == ModelVariant.MOST_FREQUENT_SENSE

    def test_word_without_wordnet_senses(self, mock_backend: MockEmbeddingBackend) -> None:
        """Test disambiguation of a word not in WordNet."""
        result = disambiguate(
            sentence="The xyzzy flibbered.",
            target_word="xyzzy",
            occurrence_index=0,
            model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
            embedding_backend=mock_backend,
        )

        assert result.success is False
        assert result.error_message is not None
        assert "No WordNet senses" in result.error_message
        assert result.ranked_candidates == []

    def test_target_word_not_in_sentence(self, mock_backend: MockEmbeddingBackend) -> None:
        """Test error when target word is not in sentence."""
        result = disambiguate(
            sentence="The cat sat on the mat.",
            target_word="dog",
            occurrence_index=0,
            model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
            embedding_backend=mock_backend,
        )
        assert result.success is False
        assert result.error_message is not None
        assert "not found" in result.error_message

    def test_prediction_result_structure(self, mock_backend: MockEmbeddingBackend) -> None:
        """Test that prediction result has correct structure."""
        result = disambiguate(
            sentence="I deposited money in the bank.",
            target_word="bank",
            occurrence_index=0,
            model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
            embedding_backend=mock_backend,
            context_strategy=ContextStrategy.LOCAL_WINDOW,
        )

        assert result.success is True
        assert len(result.ranked_candidates) > 0
        assert result.ranked_candidates[0].rank == 1
        assert result.predicted_synset_id == result.ranked_candidates[0].synset_id
        assert result.top_similarity_score is not None
        assert -1.0 <= result.top_similarity_score <= 1.0

    def test_ranked_candidates_sorted(self, mock_backend: MockEmbeddingBackend) -> None:
        """Test that ranked candidates are sorted by score."""
        result = disambiguate(
            sentence="The bat flew out of the cave.",
            target_word="bat",
            occurrence_index=0,
            model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
            embedding_backend=mock_backend,
        )

        if result.success and len(result.ranked_candidates) > 1:
            scores = [c.similarity_score for c in result.ranked_candidates]
            assert scores == sorted(scores, reverse=True)

    def test_ranked_candidates_have_correct_fields(
        self, mock_backend: MockEmbeddingBackend
    ) -> None:
        """Test that ranked candidates have all required fields."""
        result = disambiguate(
            sentence="I deposited money in the bank.",
            target_word="bank",
            occurrence_index=0,
            model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
            embedding_backend=mock_backend,
        )

        if result.success and result.ranked_candidates:
            candidate = result.ranked_candidates[0]
            assert hasattr(candidate, "rank")
            assert hasattr(candidate, "synset_id")
            assert hasattr(candidate, "definition")
            assert hasattr(candidate, "similarity_score")
            assert hasattr(candidate, "examples")
            assert hasattr(candidate, "warnings")


class TestDisambiguateFailureCases:
    """Tests for failure cases in disambiguation."""

    def test_empty_context_fails_gracefully(self) -> None:
        """Test that empty context returns failure gracefully."""
        mock = MockEmbeddingBackend()
        mock.return_zero_vector = True

        result = disambiguate(
            sentence="bank",
            target_word="bank",
            occurrence_index=0,
            model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
            embedding_backend=mock,
        )

        # Should either succeed with warning or fail gracefully
        assert isinstance(result, PredictionResult)

    def test_prediction_contains_explanation_steps(self) -> None:
        """Test that prediction result can generate explanation steps."""
        mock = MockEmbeddingBackend()

        result = disambiguate(
            sentence="I deposited money in the bank.",
            target_word="bank",
            occurrence_index=0,
            model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
            embedding_backend=mock,
        )

        if result.success:
            steps = result.get_explanation_steps()
            assert len(steps) > 0
            assert steps[0][0] == "Candidate senses"
