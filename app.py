#!/usr/bin/env python3
"""ContextSense AI - Streamlit Web Interface.

Run with: uv run streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from contextsense_ai.config import settings
from contextsense_ai.context_extractor import ContextExtractor
from contextsense_ai.disambiguation.engine import disambiguate
from contextsense_ai.embeddings.sentence_transformer import SentenceTransformerBackend
from contextsense_ai.embeddings.word2vec import Word2VecBackend
from contextsense_ai.exceptions import ContextSenseError
from contextsense_ai.schemas import (
    ContextStrategy,
    ModelVariant,
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
    st.session_state.sentence_input = "I deposited money in the bank."
if "target_input" not in st.session_state:
    st.session_state.target_input = "bank"

st.set_page_config(
    page_title="ContextSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

state = st.session_state

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #3b82f6;
    }
    .warning-badge {
        background: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧠 ContextSense AI")
    st.markdown("### Word Sense Disambiguation")
    st.markdown(
        "Combine WordNet lexical knowledge with embedding-based "
        "semantic similarity to disambiguate word meanings."
    )

    st.divider()

    st.subheader("⚙️ Resources")

    st.info(
        "Required:\n"
        "- NLTK WordNet (downloads automatically)\n\n"
        "Optional:\n"
        "- Sentence Transformers model (auto-downloads on first use)\n"
        "- Word2Vec Google News model (manual download, ~3.6GB)"
    )

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
# Helper: get or load backend
# ---------------------------------------------------------------------------


def get_backend(backend_name: str):
    """Get or create an embedding backend instance."""
    if backend_name not in st.session_state.model_backends:
        if backend_name == "sentence_transformer":
            backend = SentenceTransformerBackend()
        elif backend_name == "word2vec":
            backend = Word2VecBackend()
        else:
            return None
        st.session_state.model_backends[backend_name] = backend
        st.session_state.backend_loaded[backend_name] = False
    return st.session_state.model_backends[backend_name]


def ensure_backend_loaded(backend_name: str) -> bool:
    """Load a backend if needed."""
    backend = get_backend(backend_name)
    if backend is None:
        return False
    if not st.session_state.backend_loaded.get(backend_name, False):
        try:
            backend.load()
            st.session_state.backend_loaded[backend_name] = True
            return True
        except Exception as exc:
            st.error(f"Failed to load {backend_name}: {exc}")
            return False
    return True


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
    with st.form("wsd_form"):
        st.subheader("Input")

        col1, col2 = st.columns([2, 1])
        with col1:
            # Seeded from session state so the example buttons can drive the
            # form. Streamlit pins a widget once it owns `key`, so these must
            # not also pass `value`.
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

        st.divider()

        occ_options = []
        if sentence and target_word:
            from contextsense_ai.preprocessing import (
                find_target_occurrences,
                preprocess_for_target_lookup,
            )

            try:
                preprocessed = preprocess_for_target_lookup(sentence)
                occurrences = find_target_occurrences(preprocessed, target_word)
                if occurrences:
                    occ_options = [
                        f"Occurrence {i + 1} (position {pos})" for i, pos in enumerate(occurrences)
                    ]
                else:
                    st.warning(f"'{target_word}' not found in sentence.")
            except Exception as exc:
                st.error(f"Error analyzing sentence: {exc}")

        if occ_options:
            occurrence_idx = st.selectbox(
                "Which occurrence?",
                options=list(range(len(occ_options))),
                format_func=lambda i: occ_options[i],
                index=0,
            )
        else:
            occurrence_idx = 0

        st.divider()

        col_model, col_context = st.columns(2)

        with col_model:
            st.subheader("Model")
            model_variant = st.selectbox(
                "WSD Method",
                options=list(ModelVariant),
                format_func=lambda v: v.display_name,
                # Streamlit wants the positional index, not the option value.
                index=list(ModelVariant).index(ModelVariant.SENTENCE_TRANSFORMER_WORDNET),
            )

            if model_variant.requires_embedding:
                backend_name = "sentence_transformer"
                if st.checkbox("Use Word2Vec instead (requires manual download)", value=False):
                    backend_name = "word2vec"

                backend = get_backend(backend_name)
                if backend:
                    info = backend.model_info
                    status_color = "green" if info.is_available else "orange"
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
        st.session_state.prediction = None
        st.rerun()

    # Example buttons
    st.divider()
    st.subheader("📝 Try an example")
    example_cols = st.columns(4)
    examples = [
        ("Bank: financial", "I deposited money in the bank.", "bank"),
        ("Bank: river", "The fisherman sat near the bank of the river.", "bank"),
        ("Bat: sports", "The baseball player hit the ball with a bat.", "bat"),
        ("Bat: animal", "A bat flew out of the cave.", "bat"),
        ("Mouse: device", "I moved the cursor using the mouse.", "mouse"),
        ("Mouse: animal", "The mouse ran across the kitchen floor.", "mouse"),
        ("Spring: season", "Spring brings new life to the garden.", "spring"),
        ("Spring: mechanical", "The spring in the mattress broke.", "spring"),
    ]
    for i, (label, sent, word) in enumerate(examples):
        with example_cols[i % 4]:
            if st.button(label, key=f"ex_{i}"):
                st.session_state.sentence_input = sent
                st.session_state.target_input = word
                st.rerun()

    # Process prediction
    if submitted:
        if not sentence.strip():
            st.error("Please enter a sentence.")
        elif not target_word.strip():
            st.error("Please enter a target word.")
        else:
            with st.spinner("Running disambiguation..."):
                try:
                    backend_name = (
                        "sentence_transformer" if model_variant.requires_embedding else None
                    )
                    embedding_backend = get_backend(backend_name) if backend_name else None

                    if embedding_backend and not ensure_backend_loaded(backend_name):
                        st.error(
                            "Embedding model not available. "
                            "Check the sidebar for setup instructions."
                        )
                        st.stop()

                    result = disambiguate(
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

                    st.session_state.prediction = result

                    if result.success:
                        st.success("Disambiguation complete!")
                    else:
                        st.warning(f"Could not disambiguate: {result.error_message}")

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
            top = prediction.ranked_candidates[0] if prediction.ranked_candidates else None

            head1, head2, head3 = st.columns(3)
            with head1:
                st.metric("Predicted sense", prediction.predicted_synset_id)
            with head2:
                score = prediction.top_similarity_score
                st.metric("Similarity", "n/a" if score is None else f"{score:.4f}")
            with head3:
                st.metric("Candidates considered", len(prediction.candidates))

            if top is not None:
                st.markdown(
                    f"**{prediction.predicted_synset_id}** — {top.definition}",
                    help="Similarity scores are not probabilities.",
                )

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

    with st.form("comparison_form"):
        sentence = st.text_area(
            "Sentence",
            value="The bank approved the loan application.",
            height=80,
        )
        target_word = st.text_input("Target word", value="bank")

        submitted = st.form_submit_button("🔍 Compare Models", type="primary")

    if submitted and sentence and target_word:
        from contextsense_ai.preprocessing import (
            find_target_occurrences,
            preprocess_for_target_lookup,
        )

        try:
            preprocessed = preprocess_for_target_lookup(sentence)
            occurrences = find_target_occurrences(preprocessed, target_word)
            if not occurrences:
                st.error(f"'{target_word}' not found in sentence.")
                st.stop()
            occurrence_idx = 0
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.stop()

        with st.spinner("Loading models and running comparison..."):
            try:
                backend = SentenceTransformerBackend()
                ensure_backend_loaded("sentence_transformer")
                emb_backend = get_backend("sentence_transformer")

                if emb_backend and not state["backend_loaded"].get("sentence_transformer", False):
                    emb_backend.load()
                    state["backend_loaded"]["sentence_transformer"] = True

                if emb_backend:
                    ctx_extractor = ContextExtractor(context_window=5)
                    context_tokens = ctx_extractor.extract(
                        sentence,
                        target_word,
                        occurrence_idx,
                        strategy=ContextStrategy.LOCAL_WINDOW,
                    )

                    results = {}
                    for variant in ModelVariant:
                        if variant.requires_embedding:
                            try:
                                pred = disambiguate(
                                    sentence=sentence,
                                    target_word=target_word,
                                    occurrence_index=occurrence_idx,
                                    model_variant=variant,
                                    embedding_backend=emb_backend,
                                    context_strategy=ContextStrategy.LOCAL_WINDOW,
                                )
                                results[variant] = pred
                            except Exception:
                                results[variant] = None
                        else:
                            pred = disambiguate(
                                sentence=sentence,
                                target_word=target_word,
                                occurrence_index=occurrence_idx,
                                model_variant=variant,
                                embedding_backend=None,
                                context_strategy=ContextStrategy.FULL_SENTENCE,
                            )
                            results[variant] = pred

                    st.session_state.comparison_results = results

            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

        if st.session_state.get("comparison_results"):
            results = st.session_state.comparison_results
        st.divider()

        for variant, pred in results.items():
            if pred is None:
                continue
            with st.expander(
                f"{variant.display_name}",
                expanded=(variant == ModelVariant.SENTENCE_TRANSFORMER_WORDNET),
            ):
                if pred.success:
                    st.markdown(f"**Predicted:** {pred.predicted_synset}")
                    st.markdown(f"**Definition:** {pred.predicted_definition}")
                    if pred.top_similarity_score is not None:
                        st.metric("Similarity score", f"{pred.top_similarity_score:.4f}")
                    if pred.ranked_candidates:
                        st.markdown("**Ranked candidates:**")
                        for c in pred.ranked_candidates[:5]:
                            st.markdown(
                                f"- Rank {c.rank}: {c.synset_id} — {c.definition[:80]}... "
                                f"(score: {c.similarity_score:.4f})"
                            )
                else:
                    st.error(f"Failed: {pred.error_message}")

        # Agreement analysis
        st.divider()
        st.subheader("Agreement Analysis")
        successful_preds = {v: p for v, p in results.items() if p is not None and p.success}
        if len(successful_preds) >= 2:
            synsets = [p.predicted_synset_id for p in successful_preds.values()]
            unique = set(synsets)
            agreement_pct = (
                max(sum(1 for s in synsets if s == u) for u in unique) / len(synsets)
            ) * 100
            st.metric("Model agreement", f"{agreement_pct:.1f}%")
            st.info(
                "Agreement between models does not guarantee correctness. "
                "Different methods may make the same error for different reasons."
            )
        else:
            st.warning("Not enough successful predictions to compare.")

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
                    backend = SentenceTransformerBackend()
                    if not ensure_backend_loaded("sentence_transformer"):
                        st.error("Sentence Transformer model not available.")
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

        if st.session_state.get("evaluation_results"):
            eval_result = st.session_state.evaluation_results
            st.divider()

            metrics_tab, details_tab = st.tabs(["Metrics", "Prediction Details"])

            with metrics_tab:
                st.subheader("Model Performance")
                for variant, metrics in eval_result.per_model_results.items():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{metrics.accuracy:.2%}")
                    with col2:
                        st.metric("Precision (macro)", f"{metrics.precision_macro:.3f}")
                    with col3:
                        st.metric("Recall (macro)", f"{metrics.recall_macro:.3f}")
                    with col4:
                        st.metric("F1 (macro)", f"{metrics.f1_macro:.3f}")

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
                if eval_result.predictions:
                    import pandas as pd

                    rows = []
                    for ex_id, pred in eval_result.predictions.items():
                        row = {
                            "ID": ex_id,
                            "Sentence": pred.sentence if pred else "",
                            "Target": pred.target_word if pred else "",
                            "Gold": "",
                            "Predicted": pred.predicted_synset_id if pred else "",
                            "Correct?": "",
                            "Score": f"{pred.top_similarity_score:.4f}"
                            if pred and pred.top_similarity_score
                            else "N/A",
                            "Success": "✓" if pred and pred.success else "✗" if pred else "N/A",
                        }
                        for ex in eval_result.examples:
                            if ex.id == ex_id:
                                row["Gold"] = ex.gold_synset
                                if pred and pred.success:
                                    row["Correct?"] = (
                                        "✓ Yes"
                                        if ex.gold_synset == pred.predicted_synset_id
                                        else "✗ No"
                                    )
                                break
                        rows.append(row)

                    df = pd.DataFrame(rows)
                    st.dataframe(df, hide_index=True, use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download predictions as CSV",
                        csv,
                        "contextsense_predictions.csv",
                        "text/csv",
                        key="download_pred_csv",
                    )

    with eval_tab2:
        st.subheader("Upload Your Own Dataset")
        st.markdown(
            "Upload a CSV with columns: **sentence**, **target_word**, **gold_synset**.\n"
            "Optional: **target_occurrence**, **pos**.\n\n"
            "Example format:\n"
            "| sentence | target_word | gold_synset | target_occurrence |\n"
            "|----------|-------------|-------------|-------------------|\n"
            "| The bank approved the loan. | bank | bank.n.01 | 0 |"
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
                            backend = SentenceTransformerBackend()
                            if not ensure_backend_loaded("sentence_transformer"):
                                st.error("Model not available.")
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
                            st.session_state.evaluation_results = result
                            st.success("Evaluation complete!")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Evaluation failed: {exc}")
            except Exception as exc:
                st.error(f"Failed to load dataset: {exc}")

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

        st.markdown(
            """
            ```mermaid
            flowchart LR
                A[Input Sentence] --> B[Tokenization]
                B --> C[Target Word Lookup]
                C --> D[WordNet: Get Candidate Senses]
                D --> E[Context Extraction]
                E --> F[Embedding Model]
                F --> G[Context Vector]
                D --> H[ sense Description]
                H --> F
                F --> I[Sense Vectors]
                G --> J[Cosine Similarity]
                I --> J
                J --> K[Rank Candidates]
                K --> L[Select Top Sense]
            ```
            """
        )

    with about_tabs[2]:
        st.markdown(
            """
            ## WordNet and Synsets

            **WordNet** is a lexical database that groups English words into
            sets of synonyms called **synsets**.

            ### What is a Synset?

            A synset is a set of synonyms that represent a single distinct concept.

            Example: The synset **bank.n.01** contains:
            - bank
            - banking_instiution
            - financial_institution

            ### Synset ID Format

            WordNet synsets use the format: `lemma.pos.offset`

            - **lemma**: A representative word for the sense
            - **pos**: Part of speech (n=noun, v=verb, a=adjective, r=adverb)
            - **offset**: A unique numeric identifier

            Example: `bank.n.01` = the first noun sense of "bank"

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
            Our sample dataset is small and for demonstration only.
            SemCor and other benchmark datasets would provide more reliable evaluation.

            ### 5. Baseline Limitations
            - First-sense baseline doesn't use context
            - Most-frequent-sense baseline uses approximate frequency data

            ## Future Improvements

            1. **Better context extraction**: Use dependency parsing to find
              semantically relevant context words
            2. **Sense embeddings**: Train dedicated sense embeddings rather than using descriptions
            3. **Fine-tuned transformers**: Fine-tune a transformer on WSD tasks
            4. **Ensemble methods**: Combine multiple approaches
            5. **Larger evaluation**: Evaluate on SemCor, SemEval, or other WSD benchmarks
            6. **POS tagging integration**: Use POS to filter candidates more accurately
            7. **Domain adaptation**: Adjust for specific domains (medical, legal, etc.)
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
