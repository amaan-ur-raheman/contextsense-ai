"""The Dataset Evaluation page: score the methods on labelled data."""

from __future__ import annotations

import streamlit as st

from contextsense_ai.schemas import ContextStrategy
from contextsense_ai.ui.components import render_evaluation
from contextsense_ai.ui.state import ensure_backend_loaded, get_backend


def render_evaluation_page() -> None:
    """Render the dataset evaluation page."""
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
