"""Evaluation metrics computation using scikit-learn."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from contextsense_ai.embeddings.base import EmbeddingBackend
from contextsense_ai.schemas import (
    ContextStrategy,
    DatasetExample,
    EvaluationMetrics,
    ModelVariant,
    PredictionResult,
)


def compute_metrics(
    examples: list[DatasetExample],
    predictions: dict[str, PredictionResult],
    *,
    model_variant: ModelVariant,
    # parameters for computing metrics
    require_exact_match: bool = True,
) -> EvaluationMetrics:
    """Compute evaluation metrics for a model on a dataset.

    Args:
        examples: List of dataset examples with gold labels.
        predictions: Dict mapping example IDs to PredictionResults.
        model_variant: The model variant being evaluated.
        require_exact_match: Whether to require exact synset match.

    Returns:
        EvaluationMetrics with computed scores.
    """
    y_true: list[str] = []
    y_pred: list[str] = []
    invalid_count = 0
    skipped_count = 0

    for example in examples:
        if not example.is_valid:
            invalid_count += 1
            continue

        pred = predictions.get(example.id)
        if pred is None or not pred.success:
            skipped_count += 1
            continue

        gold = example.gold_synset
        predicted = pred.predicted_synset_id

        if require_exact_match:
            y_true.append(gold)
            y_pred.append(predicted)
        else:
            # Relaxed matching collapses both sides onto the synset's identity
            # (POS + offset), ignoring which lemma keyed it. A relaxed hit is
            # credited by emitting the gold label so the confusion matrix
            # stays meaningful.
            y_true.append(gold)
            y_pred.append(gold if _synset_match(gold, predicted) else predicted)

    if not y_true:
        return EvaluationMetrics(
            accuracy=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
            f1_weighted=0.0,
            examples_evaluated=0,
            examples_skipped=skipped_count,
            examples_invalid=invalid_count,
            per_class_metrics={},
            confusion_matrix=None,
            labels=[],
        )

    labels = sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(y_true, y_pred, normalize=True)

    try:
        precision_macro = precision_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0.0
        )
    except Exception:
        precision_macro = 0.0

    try:
        recall_macro = recall_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0.0
        )
    except Exception:
        recall_macro = 0.0

    try:
        f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0.0)
    except Exception:
        f1_macro = 0.0

    try:
        f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0.0)
    except Exception:
        f1_weighted = 0.0

    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        mask = np.array(y_true) == label
        if mask.sum() == 0:
            per_class[label] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            continue
        try:
            p = precision_score(y_true, y_pred, average=None, labels=[label], zero_division=0.0)
            r = recall_score(y_true, y_pred, average=None, labels=[label], zero_division=0.0)
            f = f1_score(y_true, y_pred, average=None, labels=[label], zero_division=0.0)
            per_class[label] = {
                "precision": float(p[0]) if len(p) > 0 else 0.0,
                "recall": float(r[0]) if len(r) > 0 else 0.0,
                "f1": float(f[0]) if len(f) > 0 else 0.0,
                "support": int(mask.sum()),
            }
        except Exception:
            per_class[label] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": int(mask.sum()),
            }

    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_list = cm.tolist()
    except Exception:
        cm_list = None

    return EvaluationMetrics(
        accuracy=float(accuracy),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        f1_weighted=float(f1_weighted),
        examples_evaluated=len(y_true),
        examples_skipped=skipped_count,
        examples_invalid=invalid_count,
        per_class_metrics=per_class,
        confusion_matrix=cm_list,
        labels=labels,
    )


def _synset_match(gold: str, predicted: str) -> bool:
    """Check whether a predicted sense refers to the same synset as the gold.

    String equality covers the common case. When the IDs differ, both are
    resolved through WordNet and compared by synset offset, which catches
    predictions keyed by a different lemma (e.g. gold ``bank.n.02`` versus a
    prediction of ``depository_financial_institution.n.01``).
    """
    if not gold or not predicted:
        return False
    if gold == predicted:
        return True

    try:
        from contextsense_ai.wordnet_service import WordNetService

        service = WordNetService()
        gold_syn = service.get_synset_from_id(gold)
        pred_syn = service.get_synset_from_id(predicted)
    except Exception:
        return False

    return gold_syn is not None and pred_syn is not None and gold_syn.offset() == pred_syn.offset()


def predict_example(
    example: DatasetExample,
    model_variant: ModelVariant,
    embedding_backend: EmbeddingBackend | None,
    *,
    context_strategy: ContextStrategy = ContextStrategy.LOCAL_WINDOW,
    context_window: int = 5,
    stopword_filtering: bool = False,
    lemmatize_context: bool = False,
    pos: str | None = None,
) -> PredictionResult:
    """Run disambiguation on one example, converting any error into a result.

    Evaluation must survive a single bad row, so failures are captured on the
    returned ``PredictionResult`` rather than propagated.

    The example's own ``pos`` field is used only when the caller does not
    override it; otherwise it is left to detection so that an unlabelled
    dataset is not silently constrained by a missing column.
    """
    from contextsense_ai.disambiguation.engine import disambiguate

    try:
        return disambiguate(
            sentence=example.sentence,
            target_word=example.target_word,
            occurrence_index=example.target_occurrence,
            model_variant=model_variant,
            embedding_backend=embedding_backend,
            context_strategy=context_strategy,
            context_window=context_window,
            stopword_filtering=stopword_filtering,
            lemmatize_context=lemmatize_context,
            pos=pos if pos is not None else example.pos,
        )
    except Exception as exc:
        return PredictionResult(
            sentence=example.sentence,
            target_word=example.target_word,
            selected_occurrence=example.target_occurrence,
            context=[],
            context_strategy=context_strategy,
            model_variant=model_variant,
            embedding_model_name=None,
            candidates=[],
            ranked_candidates=[],
            predicted_synset_id="",
            predicted_definition="",
            top_similarity_score=None,
            success=False,
            error_message=str(exc),
        )


def evaluate_model_on_examples(
    examples: list[DatasetExample],
    model_variant: ModelVariant,
    embedding_backend: EmbeddingBackend | None,
    *,
    context_strategy: ContextStrategy = ContextStrategy.LOCAL_WINDOW,
    context_window: int = 5,
    stopword_filtering: bool = False,
    lemmatize_context: bool = False,
) -> tuple[list[PredictionResult], EvaluationMetrics]:
    """Run evaluation on a list of examples and return predictions and metrics."""
    predictions: dict[str, PredictionResult] = {}
    prediction_list: list[PredictionResult] = []

    for example in examples:
        if not example.is_valid:
            continue
        pred = predict_example(
            example,
            model_variant,
            embedding_backend,
            context_strategy=context_strategy,
            context_window=context_window,
            stopword_filtering=stopword_filtering,
            lemmatize_context=lemmatize_context,
        )
        predictions[example.id] = pred
        prediction_list.append(pred)

    metrics = compute_metrics(examples, predictions, model_variant=model_variant)
    return (prediction_list, metrics)
