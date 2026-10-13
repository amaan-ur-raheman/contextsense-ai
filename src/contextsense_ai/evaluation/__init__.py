"""Evaluation modules for benchmarking WSD models."""

from contextsense_ai.evaluation.datasets import (
    DatasetSource,
    load_csv_dataset,
    load_sample_dataset,
    validate_dataset_schema,
)
from contextsense_ai.evaluation.experiments import (
    run_evaluation,
    run_model_comparison,
    save_results,
)
from contextsense_ai.evaluation.metrics import (
    EvaluationMetrics,
    compute_metrics,
)

__all__ = [
    "load_sample_dataset",
    "load_csv_dataset",
    "validate_dataset_schema",
    "DatasetSource",
    "compute_metrics",
    "EvaluationMetrics",
    "run_evaluation",
    "run_model_comparison",
    "save_results",
]
