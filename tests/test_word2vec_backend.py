"""Tests for the Word2Vec embedding backend.

These build a genuine file in the original C word2vec binary format rather than
mocking the loader, so format handling is genuinely exercised. The real Google
News vectors are ~1.5 GB and are never downloaded here.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import pytest

from contextsense_ai.embeddings.word2vec import Word2VecBackend
from contextsense_ai.exceptions import EmbeddingError

WORDS = ["bank", "bat", "cave", "deposit", "floor", "fly", "garden", "kitchen", "money", "mouse"]
DIMENSIONS = 8


def _write_word2vec_binary(path: Path, words: list[str], dimensions: int) -> None:
    """Write vectors in the original C word2vec binary format.

    Layout: a text ``<vocab> <dim>`` header line, then one record per word of
    the raw word bytes, a single space delimiter, and little-endian float32
    values. Gensim derives the word length from the space, so no length prefix
    belongs in the record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(f"{len(words)} {dimensions}\n".encode())
        for index, word in enumerate(words):
            handle.write(word.encode("utf-8") + b" ")
            vector = [(index + offset) / 10.0 for offset in range(dimensions)]
            handle.write(struct.pack(f"<{dimensions}f", *vector))


@pytest.fixture
def model_dir(tmp_path: Path, monkeypatch) -> Path:  # type: ignore[no-untyped-def]
    """Point settings at a temporary model directory."""
    from contextsense_ai.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path / "word2vec"


@pytest.fixture
def loaded_backend(model_dir: Path):
    """A backend with a small but genuine model loaded."""
    _write_word2vec_binary(model_dir / "mini.bin", WORDS, DIMENSIONS)
    backend = Word2VecBackend()
    backend.load()
    yield backend
    backend.release()


def test_missing_model_raises_actionable_error(model_dir: Path) -> None:
    """A missing model must explain where to put it and where to get it."""
    with pytest.raises(EmbeddingError) as excinfo:
        Word2VecBackend().load()

    message = str(excinfo.value)
    assert "not found" in message
    assert str(model_dir) in message
    assert "http" in message


def test_accepts_uncompressed_file(model_dir: Path) -> None:
    """Mirrors publish the uncompressed .bin, which must load."""
    _write_word2vec_binary(model_dir / "uncompressed.bin", WORDS, DIMENSIONS)

    backend = Word2VecBackend()
    backend.load()
    try:
        assert backend.is_available()
        assert backend.encode_tokens(["bank"]).is_valid
    finally:
        backend.release()


def test_accepts_gzipped_file(model_dir: Path) -> None:
    """The configured default is a .bin.gz name, which gensim reads directly."""
    raw = model_dir / "mini.bin"
    _write_word2vec_binary(raw, WORDS, DIMENSIONS)
    gz = model_dir / "mini.bin.gz"
    with open(raw, "rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())
    raw.unlink()

    backend = Word2VecBackend()
    backend.load()
    try:
        assert backend.is_available()
        assert backend.encode_tokens(["bank"]).is_valid
    finally:
        backend.release()


def test_encode_tokens_uses_mean_pooling(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """Known tokens produce a valid vector at the model's dimensionality."""
    result = loaded_backend.encode_tokens(["bank", "money"])

    assert result.is_valid
    assert not result.is_zero_vector
    assert result.dimensions == DIMENSIONS
    assert result.tokens_used == ["bank", "money"]


def test_encode_text_tokenizes_and_pools(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """Free text is tokenized, with stopwords and unknowns dropped."""
    result = loaded_backend.encode_text("deposited money in the bank")

    assert result.is_valid
    assert "bank" in result.tokens_used
    assert "in" not in result.tokens_used


def test_all_out_of_vocabulary_is_reported(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """An entirely unknown token list fails loudly instead of returning zeros."""
    result = loaded_backend.encode_tokens(["zzzqqq", "xylophonic"])

    assert not result.is_valid
    assert "OOV" in (result.error_message or "")


def test_empty_token_list_is_reported(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """Empty input is an error, not a zero vector."""
    result = loaded_backend.encode_tokens([])

    assert not result.is_valid


def test_release_clears_availability(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """release() must make the backend unavailable and free the model."""
    assert loaded_backend.is_available()
    loaded_backend.release()
    assert not loaded_backend.is_available()


def test_model_info_reports_availability(model_dir: Path) -> None:  # type: ignore[no-untyped-def]
    """Availability is reported as false until a model is loaded."""
    _write_word2vec_binary(model_dir / "mini.bin", WORDS, DIMENSIONS)

    backend = Word2VecBackend()
    assert backend.model_info.backend == "word2vec"
    assert not backend.model_info.is_available

    backend.load()
    try:
        assert backend.model_info.is_available
    finally:
        backend.release()
