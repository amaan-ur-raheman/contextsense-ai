"""WSD disambiguation modules."""

from contextsense_ai.disambiguation.baseline import (
    first_sense_baseline,
    most_frequent_sense_baseline,
)
from contextsense_ai.disambiguation.engine import disambiguate
from contextsense_ai.disambiguation.similarity import cosine_similarity, normalize_vector

__all__ = [
    "cosine_similarity",
    "normalize_vector",
    "first_sense_baseline",
    "most_frequent_sense_baseline",
    "disambiguate",
]
