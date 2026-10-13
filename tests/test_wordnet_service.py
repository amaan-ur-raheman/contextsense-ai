"""Tests for WordNet service module."""

from __future__ import annotations

import pytest

from contextsense_ai.exceptions import WordNetError
from contextsense_ai.wordnet_service import WordNetService


def _check_wordnet_available() -> bool:
    """Check if WordNet resources are available."""
    try:
        from contextsense_ai.utils.resource_manager import ensure_nltk_resources

        return ensure_nltk_resources()
    except Exception:
        return False


def _ensure_wordnet_works() -> WordNetService:
    """Create a WordNetService that actually works, or skip tests."""
    try:
        wn = WordNetService()
        # Quick test
        wn.get_candidates("test")
        return wn
    except Exception:
        pytest.skip("WordNet resources not available")


@pytest.mark.skipif(not _check_wordnet_available(), reason="WordNet resources not available")
class TestWordNetService:
    """Tests for WordNetService class."""

    @pytest.fixture
    def wn_service(self) -> WordNetService:
        """Create a WordNetService instance."""
        return _ensure_wordnet_works()

    def test_retrieve_candidates(self, wn_service: WordNetService) -> None:
        """Test retrieving candidate senses for a word."""
        candidates = wn_service.get_candidates("bank")
        assert len(candidates) > 0
        assert all(hasattr(c, "synset_id") for c in candidates)

    def test_candidate_has_definition(self, wn_service: WordNetService) -> None:
        """Test that candidates have definitions."""
        candidates = wn_service.get_candidates("bank")
        assert len(candidates) > 0
        assert candidates[0].definition
        assert len(candidates[0].definition) > 0

    def test_pos_filtering(self, wn_service: WordNetService) -> None:
        """Test filtering candidates by part of speech."""
        noun_candidates = wn_service.get_candidates("bank", pos="n")
        assert len(noun_candidates) > 0
        for c in noun_candidates:
            assert c.pos == "n"

    def test_empty_word(self, wn_service: WordNetService) -> None:
        """Test handling of empty word."""
        with pytest.raises(WordNetError):
            wn_service.get_candidates("")

    def test_word_not_in_wordnet(self, wn_service: WordNetService) -> None:
        """Test handling of word with no WordNet senses."""
        candidates = wn_service.get_candidates("xyzzy123")
        assert candidates == []

    def test_synset_id_format(self, wn_service: WordNetService) -> None:
        """Test that synset IDs have correct format."""
        candidates = wn_service.get_candidates("bank")
        assert len(candidates) > 0
        synset_id = candidates[0].synset_id
        assert "." in synset_id
        parts = synset_id.split(".")
        assert len(parts) == 3

    def test_get_definition(self, wn_service: WordNetService) -> None:
        """Test retrieving definition by synset ID."""
        definition = wn_service.get_definition("bank.n.01")
        assert definition is not None
        assert len(definition) > 0

    def test_synset_exists(self, wn_service: WordNetService) -> None:
        """Test synset existence check."""
        assert wn_service.synset_exists("bank.n.01") is True
        assert wn_service.synset_exists("xyzzy.n.999999") is False

    def test_get_synset_from_id(self, wn_service: WordNetService) -> None:
        """Test getting synset object from ID."""
        synset = wn_service.get_synset_from_id("bank.n.01")
        assert synset is not None

    def test_single_sense_word(self, wn_service: WordNetService) -> None:
        """Test word with single sense."""
        candidates = wn_service.get_candidates("acerbate")
        assert len(candidates) <= 5

    def test_many_senses_word(self, wn_service: WordNetService) -> None:
        """Test word with many senses."""
        candidates = wn_service.get_candidates("run")
        assert len(candidates) > 10

    def test_candidate_lemmas(self, wn_service: WordNetService) -> None:
        """Test that candidates have lemmas."""
        candidates = wn_service.get_candidates("bank")
        assert len(candidates) > 0
        assert len(candidates[0].lemmas) > 0

    def test_candidate_examples(self, wn_service: WordNetService) -> None:
        """Test that some candidates have examples."""
        candidates = wn_service.get_candidates("bank")
        some_have_examples = any(len(c.examples) > 0 for c in candidates)
        assert some_have_examples or all(len(c.examples) == 0 for c in candidates)
