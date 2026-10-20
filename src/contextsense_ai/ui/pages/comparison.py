"""The Model Comparison page: run one input through every method."""

from __future__ import annotations

import streamlit as st

from contextsense_ai.disambiguation.engine import disambiguate
from contextsense_ai.schemas import ContextStrategy, ModelVariant, PredictionResult
from contextsense_ai.ui.components import pick_occurrence, render_sense_chips
from contextsense_ai.ui.state import (
    backend_for_variant,
    ensure_backend_loaded,
    word2vec_installed,
)


def render_comparison() -> None:
    """Render the side-by-side model comparison."""
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
                        absent = required_name == "word2vec" and not word2vec_installed()
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
    stored: dict[ModelVariant, PredictionResult] | None = st.session_state.get("comparison_results")
    if stored:
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

        successful = {v: p for v, p in stored.items() if p.success}

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
        for variant, entry in stored.items():
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
