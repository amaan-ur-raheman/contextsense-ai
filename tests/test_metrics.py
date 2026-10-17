"""Tests for evaluation metrics module."""

from __future__ import annotations

import pytest

from contextsense_ai.evaluation.metrics import EvaluationMetrics, compute_metrics
from contextsense_ai.schemas import (
    ContextStrategy,
    DatasetExample,
    ModelVariant,
    PredictionResult,
)


class TestComputeMetrics:
    """Tests for metrics computation."""

    def _make_prediction(
        self,
        sentence: str,
        target_word: str,
        gold_synset: str,
        predicted_synset: str,
        success: bool = True,
    ) -> PredictionResult:
        """Create a mock prediction result."""
        return PredictionResult(
            sentence=sentence,
            target_word=target_word,
            selected_occurrence=0,
            context=[],
            context_strategy=ContextStrategy.FULL_SENTENCE,
            model_variant=ModelVariant.FIRST_SENSE,
            embedding_model_name=None,
            candidates=[],
            ranked_candidates=[],
            predicted_synset_id=predicted_synset,
            predicted_definition="test definition",
            top_similarity_score=0.9,
            success=success,
            warnings=[],
        )

    def _make_example(
        self,
        sentence: str,
        target_word: str,
        gold_synset: str,
        example_id: str | None = None,
    ) -> DatasetExample:
        """Create a mock dataset example."""
        return DatasetExample(
            id=example_id or f"ex_{target_word}_{gold_synset}",
            sentence=sentence,
            target_word=target_word,
            target_occurrence=0,
            gold_synset=gold_synset,
            is_valid=True,
        )

    def test_accuracy_all_correct(self) -> None:
        """Test accuracy when all predictions are correct."""
        examples = [
            self._make_example("Sent 1", "bank", "bank.n.01", "ex1"),
            self._make_example("Sent 2", "bat", "bat.n.01", "ex2"),
        ]
        predictions = {
            "ex1": self._make_prediction("Sent 1", "bank", "bank.n.01", "bank.n.01"),
            "ex2": self._make_prediction("Sent 2", "bat", "bat.n.01", "bat.n.01"),
        }

        metrics = compute_metrics(examples, predictions, model_variant=ModelVariant.FIRST_SENSE)

        assert metrics.examples_evaluated == 2
        assert metrics.accuracy == 1.0

    def test_accuracy_some_correct(self) -> None:
        """Test accuracy with mixed correctness."""
        examples = [
            self._make_example("Sent 1", "bank", "bank.n.01", "ex1"),
            self._make_example("Sent 2", "bat", "bat.n.01", "ex2"),
        ]
        predictions = {
            "ex1": self._make_prediction("Sent 1", "bank", "bank.n.01", "bank.n.01"),
            "ex2": self._make_prediction("Sent 2", "bat", "bat.n.01", "bat.n.02"),  # Wrong!
        }

        metrics = compute_metrics(examples, predictions, model_variant=ModelVariant.FIRST_SENSE)

        assert metrics.examples_evaluated == 2
        assert metrics.accuracy == 0.5

    def test_no_predictions(self) -> None:
        """Test metrics when there are no predictions."""
        examples = [
            self._make_example("Sent 1", "bank", "bank.n.01", "ex1"),
        ]
        predictions: dict[str, PredictionResult] = {}

        metrics = compute_metrics(examples, predictions, model_variant=ModelVariant.FIRST_SENSE)

        assert metrics.examples_evaluated == 0
        assert metrics.accuracy == 0.0

    def test_invalid_examples_skipped(self) -> None:
        """Test that invalid examples are skipped."""
        examples = [
            self._make_example("Sent 1", "bank", "bank.n.01", "ex1"),
            DatasetExample(
                id="ex2",
                sentence="",
                target_word="bat",
                target_occurrence=0,
                gold_synset="bat.n.01",
                is_valid=False,
                validation_error="Empty sentence",
            ),
        ]
        predictions = {
            "ex1": self._make_prediction("Sent 1", "bank", "bank.n.01", "bank.n.01"),
        }

        metrics = compute_metrics(examples, predictions, model_variant=ModelVariant.FIRST_SENSE)

        assert metrics.examples_invalid == 1
        assert metrics.examples_evaluated == 1

    def test_f1_computed(self) -> None:
        """Test that F1 score is computed."""
        examples = [
            self._make_example("Sent 1", "bank", "bank.n.01", "ex1"),
            self._make_example("Sent 2", "bat", "bat.n.01", "ex2"),
        ]
        predictions = {
            "ex1": self._make_prediction("Sent 1", "bank", "bank.n.01", "bank.n.01"),
            "ex2": self._make_prediction("Sent 2", "bat", "bat.n.01", "bat.n.02"),
        }

        metrics = compute_metrics(examples, predictions, model_variant=ModelVariant.FIRST_SENSE)

        assert metrics.f1_macro >= 0.0
        assert metrics.f1_macro <= 1.0

    def test_per_class_metrics(self) -> None:
        """Test that per-class metrics are computed."""
        examples = [
            self._make_example("Sent 1", "bank", "bank.n.01", "ex1"),
            self._make_example("Sent 2", "bank", "bank.n.01", "ex2"),
            self._make_example("Sent 3", "bat", "bat.n.01", "ex3"),
        ]
        predictions = {
            "ex1": self._make_prediction("Sent 1", "bank", "bank.n.01", "bank.n.01"),
            "ex2": self._make_prediction("Sent 2", "bank", "bank.n.01", "bank.n.01"),
            "ex3": self._make_prediction("Sent 3", "bat", "bat.n.01", "bat.n.01"),
        }

        metrics = compute_metrics(examples, predictions, model_variant=ModelVariant.FIRST_SENSE)

        assert len(metrics.per_class_metrics) > 0
        assert "bank.n.01" in metrics.per_class_metrics or "bat.n.01" in metrics.per_class_metrics


class TestEvaluationMetricsStructure:
    """Tests for EvaluationMetrics dataclass structure."""

    def test_evaluation_metrics_defaults(self) -> None:
        """Test EvaluationMetrics with default values."""
        metrics = EvaluationMetrics(
            accuracy=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
            f1_weighted=0.0,
            examples_evaluated=0,
            examples_skipped=0,
            examples_invalid=0,
        )

        assert metrics.accuracy == 0.0
        assert metrics.examples_evaluated == 0
        assert metrics.per_class_metrics == {}
        assert metrics.confusion_matrix is None
        assert metrics.labels == []


class TestMetricEdgeCases:
    """Tests for edge cases in metrics computation."""

    def _make_prediction(
        self,
        sentence: str,
        target_word: str,
        gold_synset: str,
        predicted_synset: str,
        success: bool = True,
    ) -> PredictionResult:
        """Create a mock prediction result."""
        return PredictionResult(
            sentence=sentence,
            target_word=target_word,
            selected_occurrence=0,
            context=[],
            context_strategy=ContextStrategy.FULL_SENTENCE,
            model_variant=ModelVariant.FIRST_SENSE,
            embedding_model_name=None,
            candidates=[],
            ranked_candidates=[],
            predicted_synset_id=predicted_synset,
            predicted_definition="test definition",
            top_similarity_score=0.9,
            success=success,
            warnings=[],
        )

    def test_single_class_all_correct(self) -> None:
        """Test metrics with single class and all correct."""
        examples = [
            DatasetExample(
                id="e1",
                sentence="s1",
                target_word="bank",
                target_occurrence=0,
                gold_synset="bank.n.01",
                is_valid=True,
            ),
            DatasetExample(
                id="e2",
                sentence="s2",
                target_word="bank",
                target_occurrence=0,
                gold_synset="bank.n.01",
                is_valid=True,
            ),
        ]
        predictions = {
            "e1": self._make_prediction("s1", "bank", "bank.n.01", "bank.n.01"),
            "e2": self._make_prediction("s2", "bank", "bank.n.01", "bank.n.01"),
        }

        metrics = compute_metrics(examples, predictions, model_variant=ModelVariant.FIRST_SENSE)

        assert metrics.accuracy == pytest.approx(1.0, abs=0.01)
        assert metrics.examples_evaluated == 2

    def test_single_class_all_wrong(self) -> None:
        """Test metrics with single class and all wrong."""
        examples = [
            DatasetExample(
                id="e1",
                sentence="s1",
                target_word="bank",
                target_occurrence=0,
                gold_synset="bank.n.01",
                is_valid=True,
            ),
        ]
        predictions = {
            "e1": self._make_prediction("s1", "bank", "bank.n.01", "bank.n.02"),
        }

        metrics = compute_metrics(examples, predictions, model_variant=ModelVariant.FIRST_SENSE)

        assert metrics.accuracy == pytest.approx(0.0, abs=0.01)
