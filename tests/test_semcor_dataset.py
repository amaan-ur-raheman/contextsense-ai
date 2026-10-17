"""Tests for SemCor acquisition and conversion.

The full corpus is downloaded on demand and is far too large for the test
suite, so these exercise the parts that are easy to get wrong: sense-key
resolution, the token-presence filter, and unusable-gold filtering. The WordNet
3.0 files the converter reads are not present locally either, so anything
needing them is skipped rather than faked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextsense_ai.datasets import semcor
from contextsense_ai.utils.resource_manager import ensure_nltk_resources

WORDNET_DICT_DIR = Path("data/raw/wn/dict")


def _wordnet_available() -> bool:
    try:
        return ensure_nltk_resources() and WORDNET_DICT_DIR.exists()
    except Exception:
        return False


needs_wordnet_files = pytest.mark.skipif(
    not _wordnet_available(),
    reason="WordNet 3.0 dictionary files not downloaded; run the SemCor converter",
)


def test_treebank_prefix_mapping() -> None:
    assert semcor._TREEBANK_PREFIX_TO_POS["N"] == "n"


def test_cli_defaults_are_anchored_to_project_root(monkeypatch) -> None:
    """The converter must not write into whatever directory it was launched from."""
    monkeypatch.chdir("/")
    assert semcor.DEFAULT_OUTPUT.is_absolute()
    assert semcor.DEFAULT_CACHE_DIR.is_absolute()
    assert semcor._TREEBANK_PREFIX_TO_POS["V"] == "v"
    assert semcor._TREEBANK_PREFIX_TO_POS["J"] == "a"
    assert semcor._TREEBANK_PREFIX_TO_POS["R"] == "r"


def test_useless_pos_excludes_function_words() -> None:
    """Tags that can never head a WordNet sense must be filtered."""
    for tag in ("DT", "IN", "CC", "TO", ",", ".", "None", ""):
        assert tag in semcor._USELESS_POS


def test_urls_are_https() -> None:
    """Downloads must not use plaintext transport."""
    assert semcor.WORDNET3_URL.startswith("https://")
    assert semcor.SEMCOR_BASE.startswith("https://")


@needs_wordnet_files
def test_sense_to_offset_maps_known_sense() -> None:
    """index.sense must resolve a real lemma/sense pair to a valid offset."""
    mapping = semcor.load_sense_to_offset(WORDNET_DICT_DIR)

    # artist has exactly one sense, globally numbered 18.
    assert mapping[("n", "artist", 18)] == 9812338


@needs_wordnet_files
def test_offset_to_sense_id_uses_per_lemma_numbering() -> None:
    """A sense must be numbered relative to the annotated lemma, not the head."""
    from contextsense_ai.wordnet_service import WordNetService

    mapping = semcor.build_offset_to_sense_id(WordNetService())

    # "purchase" lives inside a synset headed by "buy"; the label must still
    # be numbered among purchase's own verb senses.
    assert mapping[("purchase", 2207206)] == "purchase.v.01"
    assert mapping[("artist", 9812338)] == "artist.n.01"


@needs_wordnet_files
def test_offset_to_sense_id_matches_nltk_for_head_words() -> None:
    """For ordinary lemmas the numbering must agree with NLTK exactly."""
    from nltk.corpus import wordnet as wn

    from contextsense_ai.wordnet_service import WordNetService

    mapping = semcor.build_offset_to_sense_id(WordNetService())

    for word in ("bank", "spring", "bat"):
        synsets = wn.synsets(word)
        assert mapping[(word, synsets[0].offset())] == synsets[0].name()


def test_drop_unusable_removes_unknown_senses() -> None:
    """Rows whose gold sense does not exist must be dropped, not scored."""
    pd = pytest.importorskip("pandas")

    if not ensure_nltk_resources():
        pytest.skip("NLTK WordNet unavailable")

    frame = pd.DataFrame(
        [
            {"gold_synset": "bank.n.01", "target_word": "bank"},
            {"gold_synset": "bank.n.99", "target_word": "bank"},
        ]
    )
    cleaned = semcor.drop_unusable(frame)

    assert list(cleaned["target_word"]) == ["bank"]


def test_convert_drops_lemma_absent_from_sentence() -> None:
    """A lemma that is not a standalone token cannot be located by the engine."""
    pd = pytest.importorskip("pandas")

    frame = pd.DataFrame(
        [
            {
                "tagfile": "br-a",
                "pnum": "1",
                "snum": "1",
                "tag": "wf",
                "value": "before",
                "lemma": "be",
                "lexsn": "2:03:00::",
                "pos": "VB",
            },
        ]
    )
    path = Path("/tmp/_semcor_convert_test.csv")
    frame.to_csv(path, index=False)

    # (pos, lemma, wn_sense) resolving to a real offset, but "be" only occurs
    # inside "before", so the row must not survive.
    mapping = {("v", "be", 3): 4294414}
    examples, _ = semcor.convert(path, mapping, {("be", 4294414): "be.v.01"})
    path.unlink()

    assert examples == []
