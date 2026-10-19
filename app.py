#!/usr/bin/env python3
"""ContextSense AI - Streamlit Web Interface.

Run with: uv run streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from contextsense_ai.config import settings
from contextsense_ai.disambiguation.engine import disambiguate
from contextsense_ai.embeddings.base import EmbeddingBackend
from contextsense_ai.embeddings.sentence_transformer import SentenceTransformerBackend
from contextsense_ai.embeddings.word2vec import Word2VecBackend
from contextsense_ai.exceptions import ContextSenseError
from contextsense_ai.schemas import (
    ContextStrategy,
    ModelVariant,
    PredictionResult,
)

settings.ensure_directories()

# Initialize session state keys
if "model_backends" not in st.session_state:
    st.session_state.model_backends = {}
if "backend_loaded" not in st.session_state:
    st.session_state.backend_loaded = {}
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
# Defaults for the Playground form. The widgets own these keys, so they cannot
# also pass `value`; seeding here gives sensible first-load content and lets the
# example buttons overwrite them.
if "sentence_input" not in st.session_state:
    st.session_state.sentence_input = "She deposited the cheque at the bank."
if "target_input" not in st.session_state:
    st.session_state.target_input = "bank"

st.set_page_config(
    page_title="ContextSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Colour lives in .streamlit/config.toml. This block only handles the two
       things a config file cannot express: a spacing/hierarchy scale, and the
       interactive states Streamlit leaves half-built (focus rings especially).
       Every selector here is scoped to a class Streamlit owns, so a rename
       upstream degrades to "unstyled" rather than to something broken. */

    /* ---- scale ---------------------------------------------------------- */
    :root {
        --cs-space-1: 0.25rem;
        --cs-space-2: 0.5rem;
        --cs-space-3: 0.75rem;
        --cs-space-4: 1rem;
        --cs-space-6: 1.5rem;
        --cs-space-8: 2rem;
        /* ease-out curves, not ease-in: motion that starts slow reads as lag at
           exactly the moment the user is watching. */
        --cs-ease-out: cubic-bezier(0.23, 1, 0.32, 1);
        --cs-fast: 140ms;
        --cs-base: 180ms;
    }

    /* Streamlit renders wide containers; pull the measure in so prose stays
       inside the 65-75ch comfortable range instead of running edge to edge. */
    .block-container {
        padding-top: var(--cs-space-8);
        padding-bottom: var(--cs-space-8);
        max-width: 68rem;
    }

    /* ---- headings ------------------------------------------------------- */
    h1, h2, h3 {
        letter-spacing: -0.02em;   /* -0.04em is the floor; tighter looks cramped */
        text-wrap: balance;
    }
    h1 { line-height: 1.2; }
    p, li { text-wrap: pretty; }

    /* The emoji-glyph in a page title is decoration next to a real label.
       Dim it so it reads as a marker, not as the loudest thing on screen. */
    h1 img, h2 img, h3 img {
        opacity: 0.75;
        margin-right: 0.15em;
    }

    /* ---- buttons -------------------------------------------------------- */
    /* Give every button a press response. Without :active scale the UI feels
       like it did not hear the click. 0.97 is small enough to read as
       feedback rather than as a cartoon. */
    .stButton > button,
    .stFormSubmitButton > button {
        transition: transform var(--cs-fast) var(--cs-ease-out),
                    background-color var(--cs-fast) var(--cs-ease-out),
                    border-color var(--cs-fast) var(--cs-ease-out),
                    color var(--cs-fast) var(--cs-ease-out),
                    box-shadow var(--cs-fast) var(--cs-ease-out);
        will-change: transform;
    }
    .stButton > button:active,
    .stFormSubmitButton > button:active {
        transform: scale(0.97);
    }

    /* Hover only where a real pointer exists. On touch, hover fires on tap and
       leaves the element stuck in its hover state after the user is done. */
    @media (hover: hover) and (pointer: fine) {
        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            transform: translateY(-1px);
        }
    }

    /* Streamlit ships a weak default focus ring. Keyboard users need this to be
       unmistakably visible, so it is drawn on the element's own accent. */
    .stButton > button:focus-visible,
    .stFormSubmitButton > button:focus-visible,
    .stTextInput input:focus-visible,
    .stSelectbox [data-baseweb="select"] > div:focus-within,
    .stTextArea textarea:focus-visible {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
    }

    /* ---- sidebar -------------------------------------------------------- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--border-color);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: var(--cs-space-6);
    }

    /* ---- tabs ----------------------------------------------------------- */
    /* The active tab is identified by an underline, not by a colour shift
       alone, so the current selection survives greyscale and colour blindness. */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--primary-color);
    }
    .stTabs [data-baseweb="tab"] {
        transition: color var(--cs-base) var(--cs-ease-out),
                    background-color var(--cs-base) var(--cs-ease-out);
        padding-top: var(--cs-space-2);
        padding-bottom: var(--cs-space-2);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        font-weight: 600;
    }

    /* ---- metrics -------------------------------------------------------- */
    /* Streamlit draws a hairline above each metric. Removing it and using space
       instead stops the row reading as a table. */
    [data-testid="stMetric"] {
        border: none;
        border-top: 2px solid var(--border-color);
        padding-top: var(--cs-space-3);
        background: transparent;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.8rem;
        letter-spacing: 0.01em;
        color: var(--text-color);
        opacity: 0.62;
    }
    [data-testid="stMetricValue"] {
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.01em;
    }

    /* ---- expanders ------------------------------------------------------ */
    [data-testid="stExpander"] details {
        transition: background-color var(--cs-base) var(--cs-ease-out);
    }
    [data-testid="stExpander"] summary {
        transition: color var(--cs-fast) var(--cs-ease-out);
    }

    /* ---- app components (defined below in render helpers) --------------- */
    /* A sense chip. Rank 1 is filled with the accent because selection is the
       one thing this UI must make instantly scannable. */
    .sense-chip {
        display: inline-flex;
        align-items: baseline;
        gap: 0.4rem;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        line-height: 1.5;
        border: 1px solid var(--border-color);
        background: var(--secondary-background-color);
        color: var(--text-color);
        white-space: nowrap;
    }
    .sense-chip[data-selected="true"] {
        background: var(--primary-color);
        border-color: var(--primary-color);
        color: #ffffff;
    }
    /* Dark themes need light ink on the light-ish accent chip, not white. */
    [data-theme="dark"] .sense-chip[data-selected="true"] {
        color: #0d1116;
    }
    .sense-chip .score {
        font-variant-numeric: tabular-nums;
        opacity: 0.7;
        font-weight: 400;
    }
    [data-theme="dark"] .sense-chip[data-selected="true"] .score {
        opacity: 0.75;
    }

    /* The prediction panel. A full border plus a tinted fill -- deliberately no
       coloured side stripe, which reads as a template rather than a design. */
    .prediction-panel {
        border: 1px solid var(--border-color);
        border-radius: 12px;
        background: var(--secondary-background-color);
        padding: var(--cs-space-4) var(--cs-space-6);
        margin-bottom: var(--cs-space-4);
    }
    .prediction-panel .label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        opacity: 0.55;
        margin-bottom: var(--cs-space-1);
    }
    .prediction-panel .synset {
        font-size: 1.5rem;
        font-weight: 650;
        letter-spacing: -0.02em;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        line-height: 1.25;
    }
    .prediction-panel .gloss {
        margin-top: var(--cs-space-2);
        max-width: 68ch;
    }

    /* Reduced motion: keep the colour transitions that aid comprehension, drop
       everything that moves. Not optional -- vestibular triggers are real. */
    @media (prefers-reduced-motion: reduce) {
        .stButton > button,
        .stFormSubmitButton > button,
        .stTabs [data-baseweb="tab"],
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {
            transition-duration: 1ms !important;
        }
        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        .stButton > button:active,
        .stFormSubmitButton > button:active {
            transform: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: get or load backend
# ---------------------------------------------------------------------------


def _word2vec_installed() -> bool:
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


def pick_occurrence(sentence: str, target_word: str, *, key: str) -> int | None:
    """Ask which occurrence of ``target_word`` to disambiguate.

    Returns the 0-based index, or None when there is nothing to disambiguate.
    Shared by the Playground and Comparison pages so both resolve multi-word
    sentences identically; Comparison used to hardcode occurrence 0 and quietly
    scored the wrong token.

    Call this outside a ``st.form``: inside one the widget list is frozen until
    submit, so the options would be computed from the last submitted sentence.
    """
    if not sentence or not target_word:
        return None

    from contextsense_ai.preprocessing import (
        find_target_occurrences,
        preprocess_for_target_lookup,
    )

    try:
        occurrences = find_target_occurrences(preprocess_for_target_lookup(sentence), target_word)
    except Exception as exc:
        st.error(f"Error analyzing sentence: {exc}")
        return None

    if not occurrences:
        st.warning(f"'{target_word}' not found in sentence.")
        return None

    labels = [f"Occurrence {i + 1} (position {pos})" for i, pos in enumerate(occurrences)]
    return int(
        st.selectbox(
            "Which occurrence?",
            options=list(range(len(labels))),
            format_func=labels.__getitem__,
            index=0,
            key=key,
        )
    )


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


def render_prediction_panel(prediction: PredictionResult) -> None:
    """Render the headline prediction as a single panel.

    The answer is the reason the user opened the app, so it gets one
    unambiguous block rather than being spread across three metric tiles. The
    ranked candidates stay in a table below, where the full data lives.
    """
    score = prediction.top_similarity_score
    st.markdown(
        f"""
        <div class="prediction-panel">
          <div class="label">Selected sense</div>
          <div class="synset">{prediction.predicted_synset_id}</div>
          <div class="gloss">{prediction.predicted_definition}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Similarity", "n/a" if score is None else f"{score:.4f}")
    stat2.metric("Candidates considered", len(prediction.candidates))
    stat3.metric("Context tokens", len(prediction.context))

    st.caption(
        "A similarity score measures how closely the context and the sense "
        "description align in vector space. It is not a probability, and it is "
        "not a confidence that the prediction is correct."
    )


def render_sense_chips(candidates: list[object], limit: int = 6) -> None:
    """Render candidate senses as chips, filling the one that won.

    Rank 1 is the answer, so it is the only chip carrying the accent fill; the
    rest stay quiet. That makes the selection scannable at a glance without
    having to read the numbers.
    """
    if not candidates:
        return

    chips = "".join(
        # Lowercase literal, not str(bool): HTML attribute matching is
        # case-sensitive, so data-selected="True" misses the "true" selector
        # and the winning chip silently loses its accent fill.
        f'<span class="sense-chip" data-selected="{"true" if c.rank == 1 else "false"}">'
        f"{c.synset_id}"
        f'<span class="score">{c.similarity_score:.3f}</span>'
        f"</span>"
        for c in candidates[:limit]
    )
    remaining = len(candidates) - min(limit, len(candidates))
    suffix = f'<span class="sense-chip">+{remaining} more</span>' if remaining > 0 else ""
    st.markdown(f"{chips}{suffix}", unsafe_allow_html=True)


def render_evaluation(result: object, *, download_key: str) -> None:
    """Render metrics and per-example predictions for an EvaluationResult.

    Shared by the sample and upload tabs. The upload tab previously wrote its
    results and rendered nothing, so a completed run showed an empty page.
    """
    st.divider()
    metrics_tab, details_tab = st.tabs(["Metrics", "Prediction Details"])

    with metrics_tab:
        st.subheader("Model Performance")
        for variant, metrics in result.per_model_results.items():  # type: ignore[attr-defined]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics.accuracy:.2%}")
            col2.metric("Precision (macro)", f"{metrics.precision_macro:.3f}")
            col3.metric("Recall (macro)", f"{metrics.recall_macro:.3f}")
            col4.metric("F1 (macro)", f"{metrics.f1_macro:.3f}")

            st.divider()
            st.markdown(f"**{variant.display_name}**")
            st.write(f"Evaluated: {metrics.examples_evaluated} examples")
            st.write(f"Skipped: {metrics.examples_skipped} examples")
            st.write(f"Invalid: {metrics.examples_invalid} examples")

            if metrics.per_class_metrics:
                st.markdown("**Per-sense performance:**")
                for synset, perf in metrics.per_class_metrics.items():
                    st.markdown(
                        f"- {synset}: P={perf['precision']:.2f}, "
                        f"R={perf['recall']:.2f}, F1={perf['f1']:.2f}, "
                        f"n={perf['support']}"
                    )

    with details_tab:
        st.subheader("Prediction Details")
        predictions = result.predictions  # type: ignore[attr-defined]
        if not predictions:
            st.info("No predictions to show.")
            return

        import pandas as pd

        gold_by_id = {ex.id: ex.gold_synset for ex in result.examples}  # type: ignore[attr-defined]
        rows = []
        for ex_id, pred in predictions.items():
            score = pred.top_similarity_score
            rows.append(
                {
                    "ID": ex_id,
                    "Sentence": pred.sentence,
                    "Target": pred.target_word,
                    "Gold": gold_by_id.get(ex_id, ""),
                    "Predicted": pred.predicted_synset_id,
                    "Correct?": (
                        "✓ Yes"
                        if pred.success and gold_by_id.get(ex_id) == pred.predicted_synset_id
                        else ("✗ No" if pred.success else "n/a")
                    ),
                    # 0.0 is a real score; a truthiness test reported "N/A" for it.
                    "Score": f"{score:.4f}" if score is not None else "N/A",
                    "Success": "✓" if pred.success else "✗",
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button(
            "📥 Download predictions as CSV",
            df.to_csv(index=False).encode("utf-8"),
            "contextsense_predictions.csv",
            "text/csv",
            key=download_key,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
# Rendered after the helpers, because it calls get_backend()/_word2vec_installed()
# to report live model availability.

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
    for name, label in (
        ("sentence_transformer", "Sentence Transformers"),
        ("word2vec", "Word2Vec (Google News)"),
    ):
        backend = get_backend(name)
        info = backend.model_info if backend else None
        ready = bool(info and info.is_available)
        if ready:
            status = "ready"
        elif name == "sentence_transformer":
            status = "downloads on first use"
        else:
            status = "not installed"
        st.markdown(f"{'✅' if ready else '⬜️'} **{label}** — {status}")

    if not _word2vec_installed():
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

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

pages = {
    "🎮 Playground": "playground",
    "📊 Model Comparison": "comparison",
    "📈 Dataset Evaluation": "evaluation",
    "📚 About & Learn": "about",
}

page = st.sidebar.radio("Navigation", list(pages.keys()))

if pages[page] == "playground":
    st.title("🎮 WSD Playground")
    st.markdown(
        "Enter a sentence with an ambiguous word, select the target word, "
        "and let ContextSense AI predict its meaning."
    )

    with st.expander("📖 About Word Sense Disambiguation", expanded=False):
        st.markdown(
            """
            **What is WSD?**

            Word Sense Disambiguation is the task of determining which meaning
            (sense) of an ambiguous word is being used in a given context.

            **Example:**
            - "I deposited money in the **bank**." → financial institution
            - "The fisherman sat near the **bank** of the river." → land beside water

            **How ContextSense AI works:**

            1. **WordNet** provides candidate meanings (synsets) for the target word.
            2. **Context extraction** identifies surrounding words.
            3. **Embeddings** convert context and sense descriptions to vectors.
            4. **Cosine similarity** measures alignment between context and each sense.
            5. The sense with highest similarity is selected.
            """
        )

    # Input form
    # Sentence, target and the occurrence picker sit outside the form so they
    # rerun on every keystroke. Inside a form the widget list is frozen until
    # submit, so the occurrence options lagged one edit behind the sentence.
    st.subheader("Input")

    col1, col2 = st.columns([2, 1])
    with col1:
        # Seeded from session state so the example buttons can drive the form.
        # Streamlit pins a widget once it owns `key`, so these must not also
        # pass `value`.
        sentence = st.text_area(
            "Sentence",
            key="sentence_input",
            height=100,
            help="Enter a sentence containing the ambiguous word.",
        )
    with col2:
        target_word = st.text_input(
            "Target word",
            key="target_input",
            help="The ambiguous word to disambiguate.",
        )

    occurrence_idx = pick_occurrence(sentence, target_word, key="playground_occurrence")

    st.divider()

    with st.form("wsd_form"):
        col_model, col_context = st.columns(2)
        # Resolved from the variant inside the form; read by the handler below.
        backend_name = ""

        with col_model:
            st.subheader("Model")
            model_variant = st.selectbox(
                "WSD Method",
                options=list(ModelVariant),
                format_func=lambda v: v.display_name,
                # Streamlit wants the positional index, not the option value.
                index=list(ModelVariant).index(ModelVariant.SENTENCE_TRANSFORMER_WORDNET),
            )

            backend_name, backend = backend_for_variant(model_variant)
            if backend is not None:
                info = backend.model_info
                st.markdown(
                    f"**Model:** {info.name}\n"
                    f"**Status:** {info.status_text}\n"
                    f"**Dimensions:** {info.dimensions or '?'}D",
                )
                if not info.is_available and backend_name == "word2vec":
                    with st.expander("How to download Word2Vec model"):
                        st.code(
                            f"""# Download manually from:
{settings.word2vec_model_url}

# Then place in:
{settings.data_dir / "word2vec" / settings.default_word2vec_model}

# Or use Sentence Transformers (default, auto-downloads)."""
                        )
            else:
                st.info("This is a baseline method that does not use embeddings.")

        with col_context:
            st.subheader("Context")
            context_strategy = st.radio(
                "Strategy",
                options=list(ContextStrategy),
                format_func=lambda s: s.value.replace("_", " ").title(),
                horizontal=True,
            )

            if context_strategy == ContextStrategy.LOCAL_WINDOW:
                window = st.slider(
                    "Context window size",
                    min_value=settings.min_context_window,
                    max_value=settings.max_context_window,
                    value=settings.default_context_window,
                    help="Number of tokens on each side of the target word.",
                )
            else:
                # The window is unused for full-sentence context, but it must
                # still be valid: ContextExtractor rejects anything below the
                # configured minimum, and 0 made every submission fail.
                window = settings.default_context_window

            st.divider()
            st.subheader("Preprocessing")
            stopword_filtering = st.checkbox(
                "Filter stopwords from context",
                value=False,
                help="Remove common words like 'the', 'a', 'is' from context.",
            )
            lemmatize_context = st.checkbox(
                "Lemmatize context tokens",
                value=False,
                help="Reduce words to their base form (e.g., 'running' → 'run').",
            )

        st.divider()

        submitted = st.form_submit_button("🔍 Disambiguate", type="primary")
        cleared = st.form_submit_button("🗑️ Clear")

    if cleared:
        # Dropping the keys (rather than assigning) is what actually resets the
        # widgets; assigning to a key Streamlit already owns raises.
        for widget_key in ("sentence_input", "target_input", "playground_occurrence"):
            st.session_state.pop(widget_key, None)
        st.session_state.prediction = None
        st.rerun()

    # Example buttons
    st.divider()
    st.subheader("📝 Try an example")
    example_cols = st.columns(4)
    example_buttons = [
        # "She deposited the cheque at the bank" resolves to bank.n.02, matching
        # the label. The previously shipped "I deposited money in the bank"
        # returned bank.n.08 (a container for money at home) and so failed the
        # demo it was labelled with. The dataset in data/sample_examples.json
        # keeps the harder sentence: its gold label is verified, so the failure
        # there is real model error and must stay visible.
        ("Bank: financial", "She deposited the cheque at the bank.", "bank"),
        ("Bank: river", "The fisherman sat near the bank of the river.", "bank"),
        ("Bat: sports", "The baseball player hit the ball with a bat.", "bat"),
        ("Bat: animal", "A bat flew out of the cave.", "bat"),
        ("Mouse: device", "I moved the cursor using the mouse.", "mouse"),
        ("Mouse: animal", "The mouse ran across the kitchen floor.", "mouse"),
        ("Spring: season", "Spring brings new life to the garden.", "spring"),
        ("Spring: mechanical", "The spring in the mattress broke.", "spring"),
    ]

    # Set the inputs from a callback rather than inline. Session state for a
    # widget key cannot be assigned after that widget has been instantiated,
    # and the inputs are rendered above these buttons. on_click runs before
    # widgets are built, so the assignment is legal and no st.rerun() is needed
    # -- the rerun that follows a click already re-executes the script.
    def _load_example(sentence: str, word: str) -> None:
        st.session_state.sentence_input = sentence
        st.session_state.target_input = word
        st.session_state.prediction = None

    for i, (label, sent, word) in enumerate(example_buttons):
        with example_cols[i % 4]:
            st.button(
                label,
                key=f"ex_{i}",
                on_click=_load_example,
                args=(sent, word),
            )

    # Process prediction
    if submitted:
        if not sentence.strip():
            st.error("Please enter a sentence.")
        elif not target_word.strip():
            st.error("Please enter a target word.")
        elif occurrence_idx is None:
            st.error(f"Could not locate '{target_word}' in the sentence.")
        else:
            with st.spinner("Running disambiguation..."):
                try:
                    embedding_backend = get_backend(backend_name) if backend_name else None

                    if embedding_backend:
                        load_error = ensure_backend_loaded(backend_name)
                        if load_error:
                            st.error(
                                "Embedding model not available. "
                                f"Check the sidebar for setup instructions. ({load_error})"
                            )
                            st.stop()

                    playground_result = disambiguate(
                        sentence=sentence,
                        target_word=target_word,
                        occurrence_index=occurrence_idx,
                        model_variant=model_variant,
                        embedding_backend=embedding_backend,
                        context_strategy=context_strategy,
                        context_window=window,
                        stopword_filtering=stopword_filtering,
                        lemmatize_context=lemmatize_context,
                    )

                    st.session_state.prediction = playground_result

                    if playground_result.success:
                        st.success("Disambiguation complete!")
                    else:
                        st.warning(f"Could not disambiguate: {playground_result.error_message}")

                except ContextSenseError as exc:
                    st.error(f"Error: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    # Render the stored prediction. This reads session state rather than the
    # local `result`, so the result stays on screen across reruns (for example
    # after switching pages or changing a sidebar control).
    prediction = st.session_state.get("prediction")
    if prediction is not None:
        st.divider()
        st.subheader("🔎 Result")

        if not prediction.success:
            st.warning(f"Could not disambiguate: {prediction.error_message or 'unknown error'}")
        else:
            render_prediction_panel(prediction)

            render_sense_chips(prediction.ranked_candidates)

            with st.expander("Context used", expanded=False):
                if prediction.context:
                    st.code(" ".join(prediction.context), language=None)
                else:
                    st.info(
                        "No context tokens. The selected method is a context-free "
                        "baseline, which ignores the sentence."
                    )

            with st.expander("Ranked candidates", expanded=False):
                st.dataframe(
                    [
                        {
                            "rank": c.rank,
                            "synset": c.synset_id,
                            "pos": c.pos,
                            "score": round(c.similarity_score, 4),
                            "definition": c.definition,
                        }
                        for c in prediction.ranked_candidates
                    ],
                    width="stretch",
                    hide_index=True,
                )

            with st.expander("How this was decided", expanded=False):
                from contextsense_ai.explanation import generate_explanation

                for title, body in generate_explanation(prediction):
                    st.markdown(f"**{title}** — {body}")

            if prediction.warnings:
                with st.expander("Notes", expanded=False):
                    for warning in prediction.warnings:
                        st.markdown(f"- {warning}")

elif pages[page] == "comparison":
    st.title("📊 Model Comparison")
    st.markdown("Run the same input through multiple WSD methods and compare their predictions.")

    # Outside the form so the occurrence options track the sentence as typed.
    sentence = st.text_area(
        "Sentence",
        value="The bank approved the loan application.",
        height=80,
        key="comparison_sentence",
    )
    target_word = st.text_input("Target word", value="bank", key="comparison_target")
    occurrence_idx = pick_occurrence(sentence, target_word, key="comparison_occurrence")

    submitted = st.button("🔍 Compare Models", type="primary")

    if submitted and sentence and target_word and occurrence_idx is not None:
        with st.spinner("Loading models and running comparison..."):
            results: dict[ModelVariant, PredictionResult] = {}
            # (method, one-line reason, long detail or None). The page must say
            # which methods did not run -- silently dropping one is how a
            # three-model comparison reads as a four-model one -- but the detail
            # belongs collapsed, not as a wall of text across the page.
            skipped: list[tuple[str, str, str | None]] = []

            for variant in ModelVariant:
                required_name, emb_backend = backend_for_variant(variant)
                if variant.requires_embedding:
                    if emb_backend is None:
                        skipped.append((variant.display_name, "no backend available", None))
                        continue
                    load_error = ensure_backend_loaded(required_name)
                    if load_error:
                        # Ask the filesystem rather than pattern-matching the
                        # exception text: "not installed" is a normal state with
                        # a known fix, whereas a load failure is a real problem
                        # whose detail the user needs to see.
                        absent = required_name == "word2vec" and not _word2vec_installed()
                        skipped.append(
                            (
                                variant.display_name,
                                "model not installed" if absent else "could not load model",
                                None if absent else load_error,
                            )
                        )
                        continue

                try:
                    # Baselines ignore the context strategy entirely, so passing
                    # it uniformly keeps the comparison honest about what each
                    # method actually consumed.
                    results[variant] = disambiguate(
                        sentence=sentence,
                        target_word=target_word,
                        occurrence_index=occurrence_idx,
                        model_variant=variant,
                        embedding_backend=emb_backend if variant.requires_embedding else None,
                        context_strategy=ContextStrategy.LOCAL_WINDOW,
                    )
                except Exception as exc:
                    results[variant] = PredictionResult(
                        sentence=sentence,
                        target_word=target_word,
                        selected_occurrence=occurrence_idx,
                        context=[],
                        context_strategy=ContextStrategy.LOCAL_WINDOW,
                        model_variant=variant,
                        embedding_model_name=None,
                        candidates=[],
                        ranked_candidates=[],
                        predicted_synset_id="",
                        predicted_definition="",
                        top_similarity_score=None,
                        success=False,
                        error_message=str(exc),
                    )

            if not results:
                st.error("No model variant could be evaluated.")
                st.stop()

            st.session_state.comparison_results = results
            st.session_state.comparison_skipped = skipped

    # Rendered outside the submit branch so results survive reruns. This used to
    # sit inside it, so the table vanished on any later interaction.
    results = st.session_state.get("comparison_results")
    if results:
        skipped = st.session_state.get("comparison_skipped") or []
        if skipped:
            # One compact line per method. The instructions for getting the model
            # installed already live in the sidebar, so repeating them across the
            # page is noise; the raw error only appears when it is something other
            # than a plain "not installed".
            names = ", ".join(f"{name} ({reason})" for name, reason, _ in skipped)
            st.info(f"Ran {len(results)} of the available methods. Not evaluated: {names}.")
            details = [(name, detail) for name, _, detail in skipped if detail]
            if details:
                with st.expander("Why a method could not load"):
                    for name, detail in details:
                        st.markdown(f"**{name}** — {detail}")
            else:
                st.caption(
                    "See the sidebar for setup instructions. The Sentence "
                    "Transformers backend downloads automatically and needs no "
                    "manual step."
                )
        st.divider()

        successful = {v: p for v, p in results.items() if p.success}

        # The side-by-side table is the point of this page, so it leads. Four
        # stacked expanders each restating their own answer hid the only thing
        # the user came here to see: where the models differ.
        if successful:
            st.dataframe(
                [
                    {
                        "Method": variant.display_name,
                        "Predicted sense": p.predicted_synset_id,
                        "Similarity": (
                            "n/a"
                            if p.top_similarity_score is None
                            else round(p.top_similarity_score, 4)
                        ),
                        "Definition": p.predicted_definition,
                    }
                    for variant, p in successful.items()
                ],
                width="stretch",
                hide_index=True,
            )

            distinct = {p.predicted_synset_id for p in successful.values()}
            stat1, stat2 = st.columns(2)
            stat1.metric("Methods run", len(successful))
            stat2.metric("Distinct answers", len(distinct))
            if len(distinct) == 1:
                st.info(
                    "Every method returned the same sense here. Agreement is not "
                    "evidence of correctness — two methods can share a mistake."
                )
            elif len(distinct) < len(successful):
                st.caption(
                    f"{len(successful) - len(distinct)} method(s) matched the "
                    "majority answer. Expand any method below for its ranking."
                )

        st.subheader("Per-method detail")
        for variant, entry in results.items():
            with st.expander(f"{variant.display_name}"):
                if entry.success:
                    render_sense_chips(entry.ranked_candidates)
                    if entry.ranked_candidates:
                        st.dataframe(
                            [
                                {
                                    "rank": c.rank,
                                    "synset": c.synset_id,
                                    "pos": c.pos,
                                    "score": round(c.similarity_score, 4),
                                    "definition": c.definition,
                                }
                                for c in entry.ranked_candidates
                            ],
                            width="stretch",
                            hide_index=True,
                        )
                else:
                    st.error(f"Failed: {entry.error_message}")

elif pages[page] == "evaluation":
    st.title("📈 Dataset Evaluation")
    st.markdown(
        "Evaluate WSD model performance on labelled examples. "
        "Results show real metrics computed from actual predictions."
    )

    st.divider()

    eval_tab1, eval_tab2 = st.tabs(["Sample Dataset", "Upload CSV"])

    with eval_tab1:
        st.subheader("Built-in Sample Examples")
        st.info(
            "This is a small demonstration dataset with 8 examples. "
            "It is NOT a benchmark and should not be used to report model performance."
        )

        if st.button("Run Evaluation on Sample", type="primary"):
            with st.spinner("Loading model and evaluating..."):
                try:
                    load_error = ensure_backend_loaded("sentence_transformer")
                    if load_error:
                        st.error(f"Sentence Transformer model not available: {load_error}")
                        st.stop()

                    emb_backend = get_backend("sentence_transformer")

                    from contextsense_ai.evaluation.datasets import load_sample_dataset

                    examples = load_sample_dataset()
                    from contextsense_ai.evaluation.experiments import (
                        DEFAULT_MODEL_VARIANTS,
                        run_model_comparison,
                    )

                    result = run_model_comparison(
                        examples=examples,
                        embedding_backend=emb_backend,
                        model_variants=list(DEFAULT_MODEL_VARIANTS),
                        context_strategy=ContextStrategy.LOCAL_WINDOW,
                        context_window=5,
                        dataset_name="sample",
                    )

                    st.session_state.evaluation_results = result

                    st.success("Evaluation complete!")
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

        sample_result = st.session_state.get("evaluation_results")
        if sample_result:
            render_evaluation(sample_result, download_key="download_sample_csv")

    with eval_tab2:
        st.subheader("Upload Your Own Dataset")
        st.markdown(
            "Upload a CSV with columns: **sentence**, **target_word**, **gold_synset**.\n"
            "Optional: **target_occurrence**, **pos**.\n\n"
            "Example format:\n"
            "| sentence | target_word | gold_synset | target_occurrence |\n"
            "|----------|-------------|-------------|-------------------|\n"
            "| The bank approved the loan. | bank | bank.n.02 | 0 |"
        )

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            from contextsense_ai.evaluation.datasets import load_csv_dataset

            try:
                examples = load_csv_dataset(uploaded)
                valid = [e for e in examples if e.is_valid]
                invalid = len(examples) - len(valid)

                st.write(f"Total examples: {len(examples)}")
                st.write(f"Valid: {len(valid)}")
                st.write(f"Invalid: {invalid}")

                if invalid > 0:
                    st.warning(f"{invalid} examples have invalid data and will be skipped.")

                if st.button("Run Evaluation on Uploaded Dataset", type="primary"):
                    with st.spinner("Evaluating..."):
                        try:
                            load_error = ensure_backend_loaded("sentence_transformer")
                            if load_error:
                                st.error(f"Model not available: {load_error}")
                                st.stop()

                            emb_backend = get_backend("sentence_transformer")
                            from contextsense_ai.evaluation.experiments import (
                                DEFAULT_MODEL_VARIANTS,
                                run_model_comparison,
                            )

                            result = run_model_comparison(
                                examples=examples,
                                embedding_backend=emb_backend,
                                model_variants=list(DEFAULT_MODEL_VARIANTS),
                                context_strategy=ContextStrategy.LOCAL_WINDOW,
                                dataset_name=getattr(uploaded, "name", "uploaded.csv"),
                            )
                            # Separate key from the sample tab: both tabs wrote
                            # `evaluation_results`, so running an upload silently
                            # replaced the sample metrics shown in the other tab.
                            st.session_state.uploaded_evaluation_results = result
                            st.success("Evaluation complete!")
                        except Exception as exc:
                            st.error(f"Evaluation failed: {exc}")
            except Exception as exc:
                st.error(f"Failed to load dataset: {exc}")

        uploaded_result = st.session_state.get("uploaded_evaluation_results")
        if uploaded_result:
            render_evaluation(uploaded_result, download_key="download_upload_csv")

elif pages[page] == "about":
    st.title("📚 About & Learn")
    st.markdown(
        "Educational resources for understanding Word Sense Disambiguation "
        "and how ContextSense AI works."
    )

    about_tabs = st.tabs(
        [
            "What is WSD?",
            "How it Works",
            "WordNet & Synsets",
            "Embeddings",
            "Evaluation Metrics",
            "Limitations & Future Work",
        ]
    )

    with about_tabs[0]:
        st.markdown(
            """
            ## What is Word Sense Disambiguation?

            **Word Sense Disambiguation (WSD)** is the task of determining which
            meaning of an ambiguous word is being used in a particular context.

            ### Example

            The word **"bank"** has multiple meanings:

            | Context | Intended Meaning |
            |---------|-----------------|
            | "I deposited money in the **bank**." | Financial institution |
            | "The fisherman sat near the **bank** of the river." | Land beside water |
            | "The plane flew over the **bank**." | Cloud bank / snow bank |

            Humans disambiguate effortlessly using context. Computers need
            explicit methods to do the same.

            ### Why is WSD Hard?

            - Many words have dozens of possible senses
            - Contexts can be ambiguous themselves
            - Some senses are rare and poorly represented in data
            - Related senses can be very close in meaning
            """
        )

    with about_tabs[1]:
        st.markdown(
            """
            ## How ContextSense AI Works

            Our pipeline has 5 stages:

            ### 1. Candidate Generation (WordNet)
            For the target word, retrieve all possible meanings (synsets) from WordNet.

            ### 2. Context Extraction
            Extract surrounding words as context, excluding the target word itself.

            ### 3. Vector Representation
            - **Context vector**: Encode context words into a single vector
            - **Sense vectors**: Encode each candidate sense description into a vector

            ### 4. Similarity Calculation
            Compute cosine similarity between context vector and each sense vector.

            ### 5. Ranking and Prediction
            Rank candidates by similarity score; select the highest-ranked sense.

            ### The Cosine Similarity Formula

            ```
            cosine_similarity(A, B) = dot(A, B) / (norm(A) × norm(B))
            ```

            - Range: [-1, 1]
            - 1 = identical direction
            - 0 = orthogonal (no similarity)
            - -1 = opposite direction

            **Note:** Similarity is NOT probability. A score of 0.8 does not
            mean "80% confident this is correct."
            """
        )
        st.divider()
        st.subheader("Architecture Diagram")

        # Streamlit has no built-in Mermaid renderer, so a ```mermaid fence
        # displayed as raw source. Plain text renders everywhere.
        st.code(
            """
  sentence
      |
      v
  tokenize  ->  locate target occurrence (POS tag here too)
      |                        |
      |                        v
      |               +-------------------+
      |               | WordNet candidates|-- POS filter
      |               +-------------------+
      |                        |
      |                        v
      |                 candidate senses
      |                        |
      |        definition + examples + lemmas + hypernyms
      |                        |
      |                        v
      |                 sense description ------------+
      |                                                  |
      v                                                  v
  context tokens  (target removed, window or full) --> embedding --> cosine
                                                             --> rank
                                                             --> top sense
            """.strip(),
            language="text",
        )

    with about_tabs[2]:
        st.markdown(
            """
            ## WordNet and Synsets

            **WordNet** is a lexical database that groups English words into
            sets of synonyms called **synsets**.

            ### What is a Synset?

            A synset is a set of synonyms that represent a single distinct concept.

            Example: The synset **bank.n.02** contains:
            - bank
            - banking_concern
            - banking_company
            - depository_financial_institution

            Its gloss is "a financial institution that holds money deposits".

            ### Synset ID Format

            WordNet sense identifiers use the format: `lemma.pos.sense_number`

            - **lemma**: A representative word for the sense
            - **pos**: Part of speech (n=noun, v=verb, a=adjective, r=adverb)
            - **sense_number**: That sense's position among the lemma's senses,
              counting from 01 in the order WordNet stores them

            The numbering is per-lemma, so the financial sense of "bank" is
            `bank.n.02`, not `bank.n.01`. WordNet stores the landform sense
            ("sloping land beside a body of water") first.

            ### Semantic Relations

            WordNet encodes relationships between synsets:

            - **Hypernyms**: More general concepts ("vehicle" is a hypernym of "car")
            - **Hyponyms**: More specific concepts ("car" is a hyponym of "vehicle")
            - **Meronyms**: Part-of relationships ("wheel" is a meronym of "car")
            - **Holonyms**: Has-part relationships ("car" is a holonym of "wheel")
            """
        )

    with about_tabs[3]:
        st.markdown(
            """
            ## Word Embeddings

            **Embeddings** are dense numerical vectors that represent words or
            texts in a continuous vector space where similar meanings are close together.

            ### One-Hot vs. Dense Embeddings

            | One-Hot Encoding | Dense Embeddings |
            |------------------|------------------|
            | Sparse, high-dimensional | Dense, lower-dimensional |
            | Each word is orthogonal | Similar words have similar vectors |
            | No notion of similarity | Captures semantic relationships |
            | Size = vocabulary size | Size = embedding dimension |

            ### Word2Vec

            Word2Vec (Mikolov et al., 2013) learns word embeddings by predicting
            a word from its context (CBOW) or predicting context from a word (Skip-gram).

            - **CBOW**: Predict target word from surrounding context
            - **Skip-gram**: Predict surrounding words from target word

            Word2Vec produces **static** embeddings: the same word always gets the
            same vector regardless of context.

            ### Sentence Transformers

            Sentence Transformers (Reimers & Gurevych, 2019) extend transformer
            models to produce embeddings for full sentences or passages.

            - Based on models like BERT, RoBERTa
            - Produces **contextual** embeddings
            - Better at capturing sentence-level semantics

            ### Mean Pooling

            When we have multiple word vectors (e.g., for a sentence), we can
            combine them with **mean pooling**: average all vectors to get one vector.

            ```
            context_vector = (vec_1 + vec_2 + ... + vec_n) / n
            ```

            **Limitations of mean pooling:**
            - Loses word order
            - Treats all words equally
            - Cannot model complex interactions
            """
        )

    with about_tabs[4]:
        st.markdown(
            """
            ## Evaluation Metrics

            ### Accuracy
            The fraction of correct predictions:
            ```
            accuracy = correct_predictions / total_predictions
            ```

            ### Precision, Recall, F1

            For each sense/class:

            - **Precision**: Of all times we predicted this sense, how often were we right?
            - **Recall**: Of all times this sense was correct, how often did we predict it?
            - **F1**: Harmonic mean of precision and recall

            ### Macro vs. Weighted Averaging

            - **Macro**: Average metrics across all classes, treating each equally
            - **Weighted**: Average weighted by class frequency

            ### Why Multiple Metrics?

            Accuracy alone can be misleading:
            - A model that always picks the most frequent sense may have high accuracy
            - But it fails on less common senses
            - Per-class metrics reveal these weaknesses

            ### What We Report

            - **Accuracy**: Overall correctness
            - **Precision (macro)**: Average precision across senses
            - **Recall (macro)**: Average recall across senses
            - **F1 (macro)**: Balanced measure across all senses
            - **F1 (weighted)**: F1 weighted by sense frequency
            - **Per-class metrics**: Performance for each individual sense
            """
        )

    with about_tabs[5]:
        st.markdown(
            """
            ## Limitations

            ### 1. Context Window Limitations
            Our local window context extraction captures nearby words but misses:
            - Long-range dependencies
            - Syntactic relationships
            - Discourse-level context

            ### 2. Sense Description Quality
            WordNet definitions may not be sufficient to distinguish closely related senses.
            Some senses lack example sentences.

            ### 3. Embedding Model Limitations
            - **Word2Vec**: Static embeddings, no context sensitivity,
              mean pooling loses information
            - **Sentence Transformers**: Better but still may not capture
              fine-grained sense distinctions

            ### 4. Evaluation Dataset
            The bundled 8-example sample is for demonstration only. The SemCor
            conversion and evaluation code ship here; the corpus itself is
            fetched on demand. Run `uv run run-evaluation` to reproduce, and see
            the committed results in `docs/evaluation.md`.

            ### 5. Baseline Limitations
            - First-sense baseline doesn't use context
            - Most-frequent-sense baseline uses approximate frequency data

            ## Future Improvements

            1. **Better context extraction**: Use dependency parsing to find
              semantically relevant context words
            2. **Sense embeddings**: Train dedicated sense embeddings rather than using descriptions
            3. **Fine-tuned transformers**: Fine-tune a transformer on WSD tasks
            4. **Ensemble methods**: Combine multiple approaches
            5. **Wider benchmarks**: Evaluate on SemEval or other WSD benchmarks
            6. **Domain adaptation**: Adjust for specific domains (medical, legal, etc.)
            """
        )

    st.divider()
    st.subheader("References")
    st.markdown(
        """
        - Miller, G. A. (1995). WordNet: A lexical database for English.
          *Communications of the ACM*, 38(11), 39-41.
        - Mikolov, T., et al. (2013). Efficient estimation of word representations
          in vector space. *arXiv:1301.3781*.
        - Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings
          using Siamese BERT-networks. *EMNLP-IJCNLP 2019*.
        - Navigli, R. (2009). Word sense disambiguation: A survey.
          *ACM Computing Surveys*, 41(2), 1-69.
        """
    )
