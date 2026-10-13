"""
ContextSense AI - Context-aware Word Sense Disambiguation.

This package provides a complete WSD system combining WordNet lexical knowledge
with embedding-based semantic similarity.
"""

__version__ = "0.1.0"

from contextsense_ai.config import settings
from contextsense_ai.exceptions import (
    ContextSenseError,
    DisambiguationError,
    EmbeddingError,
    EvaluationError,
    PreprocessingError,
    WordNetError,
)
from contextsense_ai.schemas import (
    CandidateSense,
    DatasetExample,
    EvaluationResult,
    ModelVariant,
    PredictionResult,
    RankedCandidate,
)

__all__ = [
    "settings",
    "ContextSenseError",
    "WordNetError",
    "EmbeddingError",
    "DisambiguationError",
    "EvaluationError",
    "PreprocessingError",
    "CandidateSense",
    "RankedCandidate",
    "PredictionResult",
    "ModelVariant",
    "EvaluationResult",
    "DatasetExample",
]
