"""The Playground page: disambiguate one sentence interactively."""

from __future__ import annotations

import streamlit as st

from contextsense_ai.config import settings
from contextsense_ai.disambiguation.engine import disambiguate
from contextsense_ai.exceptions import ContextSenseError
from contextsense_ai.schemas import ContextStrategy, ModelVariant
from contextsense_ai.ui.components import (
    pick_occurrence,
    render_prediction_panel,
    render_sense_chips,
)
from contextsense_ai.ui.state import backend_for_variant, ensure_backend_loaded, get_backend


def render_playground() -> None:
    """Render the interactive disambiguation playground."""
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
            # Streamlit types selectbox as returning Optional. With a
            # non-empty options list and a default index it always has a
            # value, but the guard is what narrows the type -- and a widget
            # that somehow returned nothing must not silently become a
            # baseline.
            if model_variant is None:
                st.stop()

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
            # Same reasoning as the method selectbox: Streamlit types radio as
            # Optional, and this narrows it.
            if context_strategy is None:
                st.stop()

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
