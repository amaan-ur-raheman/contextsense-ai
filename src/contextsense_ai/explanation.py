"""Explanation generation for WSD predictions."""

from __future__ import annotations

from contextsense_ai.schemas import PredictionResult


def generate_explanation(result: PredictionResult) -> list[tuple[str, str]]:
    """Generate a step-by-step explanation of the disambiguation result.

    Returns a list of (step_title, step_content) tuples.
    """
    if not result.success:
        return [
            ("Error", result.error_message or "Disambiguation failed."),
        ]

    steps: list[tuple[str, str]] = []

    steps.append(
        (
            "Candidate senses",
            f"Found {len(result.candidates)} possible meaning(s) for "
            f"'{result.target_word}' in WordNet.",
        )
    )

    steps.append(
        (
            "Context strategy",
            f"Using '{result.context_strategy.value}' strategy.",
        )
    )

    if result.context:
        ctx_display = ", ".join(result.context[:15])
        if len(result.context) > 15:
            ctx_display += "..."
        steps.append(("Context used", f"Tokens: {ctx_display}"))

    if result.embedding_model_name:
        steps.append(
            (
                "Embedding model",
                f"Using '{result.embedding_model_name}' to encode context and sense descriptions.",
            )
        )
        steps.append(
            (
                "Representation",
                "Context tokens and sense descriptions are converted to vectors "
                "using the embedding model.",
            )
        )
    else:
        steps.append(
            (
                "Method",
                "Baseline method used (no context or embeddings required).",
            )
        )

    steps.append(
        (
            "Similarity calculation",
            "Cosine similarity computed between context vector and each candidate sense vector.",
        )
    )

    if result.ranked_candidates:
        top = result.ranked_candidates[0]
        steps.append(
            (
                "Selected sense",
                f"**{top.synset_id}** — {top.definition}\n"
                f"Score: {top.similarity_score:.4f} (rank 1 of {len(result.ranked_candidates)})",
            )
        )

        if len(result.ranked_candidates) > 1:
            steps.append(
                (
                    "Other candidates",
                    "\n".join(
                        f"Rank {c.rank}: {c.synset_id} — {c.definition[:60]}... "
                        f"(score: {c.similarity_score:.4f})"
                        for c in result.ranked_candidates[1:6]
                    ),
                )
            )

    for warning in result.warnings:
        steps.append(("Note", warning))

    return steps


def format_similarity_note() -> str:
    """Return a note about interpreting similarity scores."""
    return (
        "Similarity scores indicate how closely the context vector aligns with each "
        "candidate sense vector. They are NOT probabilities and do NOT represent "
        "confidence that the prediction is correct."
    )
