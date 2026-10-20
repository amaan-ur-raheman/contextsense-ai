"""Experiment running and result export for WSD evaluation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from contextsense_ai.embeddings.base import EmbeddingBackend, backend_serves_variant
from contextsense_ai.exceptions import EvaluationError
from contextsense_ai.schemas import (
    ContextStrategy,
    DatasetExample,
    EvaluationMetrics,
    EvaluationResult,
    ModelVariant,
    PredictionResult,
)

DEFAULT_MODEL_VARIANTS = [
    ModelVariant.FIRST_SENSE,
    ModelVariant.MOST_FREQUENT_SENSE,
    ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
]


def _config(
    *,
    context_strategy: ContextStrategy,
    context_window: int,
    stopword_filtering: bool,
    lemmatize_context: bool,
    embedding_backend: EmbeddingBackend | None,
    examples: list[DatasetExample],
) -> dict[str, object]:
    """Build the reproducibility record stored alongside every result."""
    return {
        "context_strategy": context_strategy.value,
        "context_window": context_window,
        "stopword_filtering": stopword_filtering,
        "lemmatize_context": lemmatize_context,
        "embedding_model": embedding_backend.model_info.name if embedding_backend else None,
        "total_examples": len(examples),
        "valid_examples": sum(1 for e in examples if e.is_valid),
    }


def run_evaluation(
    examples: list[DatasetExample],
    model_variant: ModelVariant,
    embedding_backend: EmbeddingBackend | None = None,
    *,
    context_strategy: ContextStrategy = ContextStrategy.LOCAL_WINDOW,
    context_window: int = 5,
    stopword_filtering: bool = False,
    lemmatize_context: bool = False,
    dataset_name: str = "evaluation",
    output_path: Path | str | None = None,
) -> EvaluationResult:
    """Run a full evaluation of one model variant on a dataset."""
    context_strategy = ContextStrategy(context_strategy)
    model_variant = ModelVariant(model_variant)

    if model_variant.requires_embedding and not backend_serves_variant(
        embedding_backend, model_variant
    ):
        # Loud, because this is a caller error: the backend cannot produce the
        # model that was asked for. Running anyway would return a valid-looking
        # result for a different experiment.
        supplied = "none" if embedding_backend is None else embedding_backend.model_info.backend
        raise EvaluationError(
            f"Model variant '{model_variant.display_name}' needs its own embedding "
            f"backend, but the one supplied ('{supplied}') does not serve it."
        )

    prediction_list, metrics = evaluate_single_variant(
        examples=examples,
        model_variant=model_variant,
        embedding_backend=embedding_backend,
        context_strategy=context_strategy,
        context_window=context_window,
        stopword_filtering=stopword_filtering,
        lemmatize_context=lemmatize_context,
    )
    # Keyed by example id, same idiom as run_model_comparison: invalid rows are
    # skipped, so the two iterators line up exactly.
    predictions = {
        example.id: pred
        for example, pred in zip(
            (e for e in examples if e.is_valid), prediction_list, strict=True
        )
    }

    result = EvaluationResult(
        dataset_name=dataset_name,
        dataset_source="local",
        examples=examples,
        per_model_results={model_variant: metrics},
        shared_subset_evaluated=metrics.examples_evaluated,
        config={
            "model_variant": model_variant.value,
            **_config(
                context_strategy=context_strategy,
                context_window=context_window,
                stopword_filtering=stopword_filtering,
                lemmatize_context=lemmatize_context,
                embedding_backend=embedding_backend,
                examples=examples,
            ),
        },
        timestamp=datetime.now(UTC).isoformat(),
        predictions=predictions,
    )

    if output_path:
        save_results(result, output_path)

    return result


def run_model_comparison(
    examples: list[DatasetExample],
    embedding_backend: EmbeddingBackend | None = None,
    *,
    model_variants: list[ModelVariant] | None = None,
    context_strategy: ContextStrategy = ContextStrategy.LOCAL_WINDOW,
    context_window: int = 5,
    stopword_filtering: bool = False,
    lemmatize_context: bool = False,
    dataset_name: str = "comparison",
    output_path: Path | str | None = None,
) -> EvaluationResult:
    """Evaluate several model variants on the same examples.

    Every variant is scored on the same subset of examples so the numbers are
    directly comparable; variants that cannot run are reported as unavailable
    rather than silently dropped.
    """
    context_strategy = ContextStrategy(context_strategy)
    model_variants = [ModelVariant(v) for v in (model_variants or DEFAULT_MODEL_VARIANTS)]

    per_model_results: dict[ModelVariant, EvaluationMetrics] = {}
    # Keyed by example id so results survive export and duplicate sentences.
    # Holds the last variant evaluated; ``predictions_by_variant`` keeps them all.
    all_predictions: dict[str, PredictionResult] = {}
    predictions_by_variant: dict[str, dict[str, PredictionResult]] = {}
    skipped_variants: list[str] = []

    for variant in model_variants:
        if variant.requires_embedding:
            # Refuse to run an embedding variant against a backend that serves
            # a different model. Substituting one silently produces valid
            # numbers for the wrong experiment -- a Word2Vec-labelled row
            # holding Sentence Transformers results.
            if embedding_backend is None:
                skipped_variants.append(variant.value)
                continue
            if not backend_serves_variant(embedding_backend, variant):
                skipped_variants.append(f"{variant.value} (wrong backend)")
                continue

        prediction_list, metrics = evaluate_single_variant(
            examples=examples,
            model_variant=variant,
            embedding_backend=embedding_backend,
            context_strategy=context_strategy,
            context_window=context_window,
            stopword_filtering=stopword_filtering,
            lemmatize_context=lemmatize_context,
        )
        per_model_results[variant] = metrics

        variant_predictions: dict[str, PredictionResult] = {}
        for example, pred in zip((e for e in examples if e.is_valid), prediction_list, strict=True):
            all_predictions[example.id] = pred
            variant_predictions[example.id] = pred
        predictions_by_variant[variant.value] = variant_predictions

    if not per_model_results:
        raise EvaluationError(
            "No model variant could be evaluated. Embedding-based variants require "
            "a loaded embedding backend."
        )

    # All variants score the same rows unless one of them failed to predict, so
    # the smallest evaluation count is the subset they all share.
    shared_subset = min(
        (m.examples_evaluated for m in per_model_results.values()),
        default=0,
    )

    result = EvaluationResult(
        dataset_name=dataset_name,
        dataset_source="local",
        examples=examples,
        per_model_results=per_model_results,
        shared_subset_evaluated=shared_subset,
        config={
            "model_variants": [v.value for v in model_variants],
            "skipped_variants": skipped_variants,
            **_config(
                context_strategy=context_strategy,
                context_window=context_window,
                stopword_filtering=stopword_filtering,
                lemmatize_context=lemmatize_context,
                embedding_backend=embedding_backend,
                examples=examples,
            ),
        },
        timestamp=datetime.now(UTC).isoformat(),
        predictions=all_predictions,
        predictions_by_variant=predictions_by_variant,
    )

    if output_path:
        save_results(result, output_path)

    return result


def evaluate_single_variant(
    examples: list[DatasetExample],
    model_variant: ModelVariant,
    embedding_backend: EmbeddingBackend | None,
    *,
    context_strategy: ContextStrategy = ContextStrategy.LOCAL_WINDOW,
    context_window: int = 5,
    stopword_filtering: bool = False,
    lemmatize_context: bool = False,
) -> tuple[list[PredictionResult], EvaluationMetrics]:
    """Evaluate a single model variant and return its predictions and metrics."""
    from contextsense_ai.evaluation.metrics import evaluate_model_on_examples

    return evaluate_model_on_examples(
        examples=examples,
        model_variant=model_variant,
        embedding_backend=embedding_backend,
        context_strategy=context_strategy,
        context_window=context_window,
        stopword_filtering=stopword_filtering,
        lemmatize_context=lemmatize_context,
    )


def save_results(result: EvaluationResult, output_path: Path | str) -> None:
    """Save evaluation results to CSV and JSON next to ``output_path``."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    _save_predictions_csv(result, output.with_suffix(".predictions.csv"))
    _save_summary_json(result, output.with_suffix(".summary.json"))


def _save_predictions_csv(result: EvaluationResult, path: Path) -> None:
    """Save per-example predictions to CSV.

    One row per (example, model variant). ``predictions`` alone cannot express
    this: it is keyed by example id, so in a multi-variant comparison each
    variant overwrote the previous one and only the last survived.
    """
    by_variant = result.predictions_by_variant or {
        result_variant: result.predictions for result_variant in ("",)
    }

    rows: list[dict[str, object]] = []
    for variant_name, predictions in by_variant.items():
        for example in result.examples:
            pred = predictions.get(example.id)
            if pred is None:
                continue
            rows.append(
                {
                    "example_id": example.id,
                    "model_variant": variant_name or pred.model_variant.value,
                    "sentence": example.sentence,
                    "target_word": example.target_word,
                    "target_occurrence": example.target_occurrence,
                    "gold_synset": example.gold_synset,
                    "predicted_synset": pred.predicted_synset_id,
                    "predicted_definition": pred.predicted_definition,
                    "top_similarity_score": pred.top_similarity_score,
                    "success": pred.success,
                    "error_message": pred.error_message,
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {len(rows)} prediction(s) across {len(by_variant)} variant(s) to {path}")


def _save_summary_json(result: EvaluationResult, path: Path) -> None:
    """Save evaluation summary to JSON."""
    summary = {
        "dataset_name": result.dataset_name,
        "dataset_source": result.dataset_source,
        "timestamp": result.timestamp,
        "config": result.config,
        "shared_subset_evaluated": result.shared_subset_evaluated,
        "per_model_results": {
            variant.value: {
                "accuracy": metrics.accuracy,
                "precision_macro": metrics.precision_macro,
                "recall_macro": metrics.recall_macro,
                "f1_macro": metrics.f1_macro,
                "f1_weighted": metrics.f1_weighted,
                "examples_evaluated": metrics.examples_evaluated,
                "examples_skipped": metrics.examples_skipped,
                "examples_invalid": metrics.examples_invalid,
                "per_class_metrics": metrics.per_class_metrics,
            }
            for variant, metrics in result.per_model_results.items()
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {path}")
