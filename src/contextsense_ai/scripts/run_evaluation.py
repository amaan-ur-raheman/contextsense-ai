#!/usr/bin/env python3
"""Run evaluation on a dataset and save results."""

from __future__ import annotations

import argparse
import sys

from contextsense_ai.config import settings
from contextsense_ai.embeddings.base import EmbeddingBackend
from contextsense_ai.embeddings.sentence_transformer import SentenceTransformerBackend
from contextsense_ai.embeddings.word2vec import Word2VecBackend
from contextsense_ai.evaluation.datasets import load_csv_dataset, load_sample_dataset
from contextsense_ai.evaluation.experiments import (
    DEFAULT_MODEL_VARIANTS,
    run_evaluation,
    run_model_comparison,
)
from contextsense_ai.exceptions import EvaluationError
from contextsense_ai.schemas import ContextStrategy, ModelVariant


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WSD evaluation for ContextSense AI.")
    parser.add_argument("--sample", action="store_true", help="Use the built-in sample dataset.")
    parser.add_argument("--csv", type=str, default=None, help="Path to a CSV dataset file.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model variant to evaluate: "
        + ", ".join(v.value for v in ModelVariant)
        + ", or 'all'.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output path for results.")
    parser.add_argument(
        "--context-window", type=int, default=5, help="Context window size for local context."
    )
    parser.add_argument(
        "--context-strategy",
        type=str,
        default=ContextStrategy.LOCAL_WINDOW.value,
        choices=[s.value for s in ContextStrategy],
        help="Context extraction strategy.",
    )
    return parser.parse_args(argv)


def _load_backend(variant: ModelVariant) -> EmbeddingBackend | None:
    """Load the embedding backend the requested variant needs.

    Each variant has its own model, so the backend is chosen from the variant
    rather than always being Sentence Transformers. Loading the wrong one and
    labelling it as the requested model would report the wrong experiment.
    The context-free baselines do not need embeddings, so an unavailable
    backend downgrades to baseline-only evaluation rather than aborting.
    """
    if variant is ModelVariant.WORD2VEC_WORDNET:
        backend: EmbeddingBackend = Word2VecBackend()
    elif variant is ModelVariant.SENTENCE_TRANSFORMER_WORDNET:
        backend = SentenceTransformerBackend()
    else:
        return None

    try:
        backend.load()
    except Exception as exc:
        print(f"! Embedding model unavailable: {exc}")
        print("  Continuing with context-free baselines only.\n")
        return None
    print(f"✓ Embedding model loaded ({backend.model_info.name}).")
    return backend


def main(argv: list[str] | None = None) -> None:
    """Run evaluation using command-line arguments."""
    args = _parse_args(argv)

    settings.ensure_directories()

    print("=" * 60)
    print("ContextSense AI - Evaluation Run")
    print("=" * 60)
    print()

    if args.csv:
        print(f"Loading CSV dataset: {args.csv}")
        try:
            examples = load_csv_dataset(args.csv)
        except EvaluationError as exc:
            # A mistyped path is a user error, not a crash. Printing a traceback
            # for it buries the message the user needs.
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        dataset_name = args.csv
    else:
        print("Loading sample dataset...")
        examples = load_sample_dataset()
        dataset_name = "sample"

    valid = [e for e in examples if e.is_valid]
    print(f"Loaded {len(examples)} examples ({len(valid)} valid).")
    if len(valid) < len(examples):
        for example in examples:
            if not example.is_valid:
                print(f"  ! {example.id}: {example.validation_error}")
    print()

    if not valid:
        print("No valid examples to evaluate.")
        sys.exit(1)

    strategy = ContextStrategy(args.context_strategy)

    print()
    print("Running evaluation...")
    print()

    if args.model == "all":
        variants = list(DEFAULT_MODEL_VARIANTS)
        print("Variants: " + ", ".join(v.display_name for v in variants))
        print()
        # Every embedding variant in this list needs Sentence Transformers;
        # the comparison refuses to run a variant against a backend that
        # serves a different model.
        backend = _load_backend(ModelVariant.SENTENCE_TRANSFORMER_WORDNET)
        result = run_model_comparison(
            examples=examples,
            embedding_backend=backend,
            model_variants=variants,
            context_strategy=strategy,
            context_window=args.context_window,
            dataset_name=dataset_name,
            output_path=args.output,
        )
    else:
        variant = ModelVariant(args.model)
        backend = _load_backend(variant)
        if variant.requires_embedding and backend is None:
            print(f"'{variant.display_name}' requires an embedding model, which is unavailable.")
            sys.exit(1)
        print(f"Evaluating: {variant.display_name}")
        print()
        result = run_evaluation(
            examples=examples,
            model_variant=variant,
            embedding_backend=backend,
            context_strategy=strategy,
            context_window=args.context_window,
            dataset_name=dataset_name,
            output_path=args.output,
        )

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    for variant, metrics in result.per_model_results.items():
        print(f"\n{variant.display_name}:")
        print(f"  Accuracy:         {metrics.accuracy:.4f}")
        print(f"  Precision (macro): {metrics.precision_macro:.4f}")
        print(f"  Recall (macro):    {metrics.recall_macro:.4f}")
        print(f"  F1 (macro):        {metrics.f1_macro:.4f}")
        print(f"  F1 (weighted):     {metrics.f1_weighted:.4f}")
        print(f"  Evaluated: {metrics.examples_evaluated}")
        print(f"  Skipped:   {metrics.examples_skipped}")
        print(f"  Invalid:   {metrics.examples_invalid}")

    if args.output:
        print(f"\nResults saved near {args.output}")

    if backend is not None:
        backend.release()


if __name__ == "__main__":
    main()
