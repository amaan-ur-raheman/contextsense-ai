"""Exception hierarchy for ContextSense AI."""

from __future__ import annotations


class ContextSenseError(Exception):
    """Base exception for all ContextSense errors."""


class PreprocessingError(ContextSenseError):
    """Error during text preprocessing."""


class WordNetError(ContextSenseError):
    """Error related to WordNet operations."""


class EmbeddingError(ContextSenseError):
    """Error related to embedding model operations."""


class DisambiguationError(ContextSenseError):
    """Error during the disambiguation process."""


class EvaluationError(ContextSenseError):
    """Error during evaluation or benchmarking."""


class ResourceError(ContextSenseError):
    """Error related to resource management or downloads."""
