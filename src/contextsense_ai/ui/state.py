"""Session state and embedding-backend lifecycle.

The backends are heavy and long-lived, so they are created once per browser
session and held in ``st.session_state``. Everything that needs to know
whether a model is *installed* or merely *loaded* asks here, because the two
are different questions: ``is_available`` is false for every backend until
the first prediction.
"""

from __future__ import annotations

from typing import cast

import streamlit as st

from contextsense_ai.config import settings
from contextsense_ai.embeddings.base import EmbeddingBackend
from contextsense_ai.embeddings.sentence_transformer import SentenceTransformerBackend
from contextsense_ai.embeddings.word2vec import Word2VecBackend
from contextsense_ai.schemas import ModelVariant


def init_state() -> None:
    """Create the session-state keys the interface relies on."""
    if "model_backends" not in st.session_state:
        st.session_state.model_backends = {}
    if "backend_loaded" not in st.session_state:
        st.session_state.backend_loaded = {}
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None
    if "comparison_skipped" not in st.session_state:
        st.session_state.comparison_skipped = []
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None
    if "uploaded_evaluation_results" not in st.session_state:
        st.session_state.uploaded_evaluation_results = None
    # Defaults for the Playground form. The widgets own these keys, so they
    # cannot also pass `value`; seeding here gives sensible first-load content
    # and lets the example buttons overwrite them.
    if "sentence_input" not in st.session_state:
        st.session_state.sentence_input = "She deposited the cheque at the bank."
    if "target_input" not in st.session_state:
        st.session_state.target_input = "bank"


def word2vec_installed() -> bool:
    """Whether a Word2Vec vectors file is present in the model directory.

    Checked on disk rather than via ``is_model_available``, which only reports a
    *loaded* model. The sidebar needs to know what is installed, before anything
    has tried to load it.
    """
    return any((settings.data_dir / "word2vec").glob("*.bin*"))


def get_backend(backend_name: str) -> EmbeddingBackend | None:
    """Get or create an embedding backend instance."""
    if backend_name not in st.session_state.model_backends:
        backend: EmbeddingBackend
        if backend_name == "sentence_transformer":
            backend = SentenceTransformerBackend()
        elif backend_name == "word2vec":
            backend = Word2VecBackend()
        else:
            return None
        st.session_state.model_backends[backend_name] = backend
        st.session_state.backend_loaded[backend_name] = False
    return cast("EmbeddingBackend | None", st.session_state.model_backends[backend_name])


def ensure_backend_loaded(backend_name: str) -> str | None:
    """Load a backend if needed.

    Returns None on success, or a message explaining the failure. It does not
    render: every caller reports the error with its own context, and rendering
    here too produced two stacked error banners for one failure.
    """
    backend = get_backend(backend_name)
    if backend is None:
        return f"Unknown embedding backend '{backend_name}'."
    if st.session_state.backend_loaded.get(backend_name, False):
        return None
    try:
        backend.load()
        st.session_state.backend_loaded[backend_name] = True
        return None
    except Exception as exc:
        return str(exc)


def backend_for_variant(variant: ModelVariant) -> tuple[str, EmbeddingBackend | None]:
    """Return the backend name a variant requires, and the backend if available.

    The variant alone decides the model. Deriving it from a separate control let
    the UI label a run "Word2Vec + WordNet" while computing Sentence Transformer
    vectors, because ``disambiguate`` only checks that a backend was supplied.
    """
    if variant is ModelVariant.WORD2VEC_WORDNET:
        name = "word2vec"
    elif variant is ModelVariant.SENTENCE_TRANSFORMER_WORDNET:
        name = "sentence_transformer"
    else:
        return "", None
    return name, get_backend(name)
