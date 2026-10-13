"""Embedding backends for ContextSense AI."""

from contextsense_ai.embeddings.base import (
    EmbeddingBackend,
    EmbeddingModelInfo,
    EmbeddingResult,
    Vector,
)
from contextsense_ai.embeddings.sentence_transformer import SentenceTransformerBackend
from contextsense_ai.embeddings.word2vec import Word2VecBackend

__all__ = [
    "EmbeddingBackend",
    "EmbeddingResult",
    "EmbeddingModelInfo",
    "Vector",
    "Word2VecBackend",
    "SentenceTransformerBackend",
]
