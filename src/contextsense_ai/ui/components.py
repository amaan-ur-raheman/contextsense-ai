"""Reusable interface components.

Widgets that appear on more than one page, or that carry a design decision
worth making once, live here rather than being copied into each page.
"""

from __future__ import annotations

import streamlit as st

from contextsense_ai.schemas import PredictionResult, RankedCandidate


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


def render_sense_chips(candidates: list[RankedCandidate], limit: int = 6) -> None:
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
