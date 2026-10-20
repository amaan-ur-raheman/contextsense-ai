"""The application sidebar.

Rendered after the state helpers because it reports live model availability,
which it asks about rather than assuming.
"""

from __future__ import annotations

import streamlit as st

from contextsense_ai.config import settings
from contextsense_ai.ui.state import get_backend, word2vec_installed


def render_sidebar() -> None:
    """Render navigation context, model status and resource controls."""
    with st.sidebar:
        st.title("🧠 ContextSense AI")
        st.markdown("### Word Sense Disambiguation")
        st.markdown(
            "Combine WordNet lexical knowledge with embedding-based "
            "semantic similarity to disambiguate word meanings."
        )

        st.divider()

        st.subheader("⚙️ Models")

        # Show live availability rather than a static list of what exists in theory.
        # A page that quietly omits a method is exactly the failure this panel
        # exists to prevent, so it has to be the place that explains the omission.
        #
        # "Installed" and "loaded" are different questions. is_available only
        # reports a loaded model, so before the first prediction every backend
        # looks absent -- and reporting Word2Vec as "not installed" when its
        # vectors are sitting on disk is exactly the kind of false statement this
        # panel is meant to eliminate.
        for name, label in (
            ("sentence_transformer", "Sentence Transformers"),
            ("word2vec", "Word2Vec (Google News)"),
        ):
            backend = get_backend(name)
            info = backend.model_info if backend else None
            loaded = bool(info and info.is_available)
            installed = name == "word2vec" and word2vec_installed()
            if loaded:
                status, ready = "ready", True
            elif installed:
                status, ready = "installed", True
            elif name == "sentence_transformer":
                status, ready = "downloads on first use", False
            else:
                status, ready = "not installed", False
            st.markdown(f"{'✅' if ready else '⬜️'} **{label}** — {status}")

        if not word2vec_installed():
            with st.expander("Install the Word2Vec model (~3.6 GB)"):
                st.markdown(
                    "It is optional. The Sentence Transformers backend covers the "
                    "same comparison and downloads itself automatically."
                )
                st.code(
                    f"# Download from:\n{settings.word2vec_model_url}\n\n"
                    f"# Then place the file in:\n{settings.data_dir / 'word2vec'}\n\n"
                    "# Any .bin or .bin.gz name works; gensim reads gzip\n"
                    "# transparently. Restart the app once it is in place."
                )

        st.divider()
        st.subheader("📦 Data")

        st.info("NLTK WordNet downloads automatically.")

        if st.button("🔄 Download NLTK Resources", type="primary"):
            from contextsense_ai.utils.resource_manager import ensure_nltk_resources

            with st.spinner("Downloading resources..."):
                ok = ensure_nltk_resources()
            if ok:
                st.success("NLTK resources ready!")
            else:
                st.error("Failed to download. Check your internet connection.")

        st.divider()
        st.caption("ContextSense AI v0.1.0")
        st.caption("An educational WSD system for NLP students and researchers.")
