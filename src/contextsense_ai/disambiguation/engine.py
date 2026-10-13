"""Main disambiguation engine combining embedding models with WordNet."""

from __future__ import annotations

import numpy as np

from contextsense_ai.context_extractor import ContextExtractor
from contextsense_ai.disambiguation.baseline import (
    first_sense_baseline,
    most_frequent_sense_baseline,
)
from contextsense_ai.disambiguation.similarity import cosine_similarity_safe
from contextsense_ai.embeddings.base import EmbeddingBackend
from contextsense_ai.exceptions import DisambiguationError, EmbeddingError
from contextsense_ai.preprocessing import tag_target_pos
from contextsense_ai.schemas import (
    CandidateSense,
    ContextStrategy,
    ModelVariant,
    PredictionResult,
)
from contextsense_ai.wordnet_service import WordNetService

# Sense descriptions are constant text, so their vectors are cached per model.
_SENSE_VECTOR_CACHE: dict[tuple[str, str], np.ndarray] = {}


def build_sense_description(candidate: CandidateSense) -> str:
    """Build a textual description of a candidate sense for embedding."""
    parts: list[str] = [candidate.definition]
    for example in candidate.examples[:3]:
        parts.append(f"Example: {example}")
    if candidate.lemmas:
        parts.append(f"Lemmas: {', '.join(candidate.lemmas[:5])}")
    if candidate.hypernyms:
        parts.append(f"Hypernyms: {', '.join(candidate.hypernyms[:3])}")
    return " ".join(parts)


def _encode_context(
    backend: EmbeddingBackend,
    context_tokens: list[str],
    *,
    strategy: ContextStrategy,
) -> np.ndarray:
    """Encode context tokens into a vector using the embedding backend."""
    if strategy == ContextStrategy.FULL_SENTENCE:
        text = " ".join(context_tokens)
        result = backend.encode_text(text)
        if not result.is_valid or result.is_zero_vector:
            raise EmbeddingError(
                f"Context encoding produced invalid vector: {result.error_message}"
            )
        return result.vector

    result = backend.encode_tokens(context_tokens)
    if not result.is_valid or result.is_zero_vector:
        raise EmbeddingError(f"Context encoding produced invalid vector: {result.error_message}")
    return result.vector


def _encode_sense(
    backend: EmbeddingBackend,
    candidate: CandidateSense,
) -> np.ndarray:
    """Encode a candidate sense description into a vector, memoised per model.

    A sense description is fixed text, so its vector never changes for a given
    embedding model. Without caching, every example re-encodes every candidate
    sense: at roughly 65 ms per call and ~24 candidate senses per word, that is
    over a second of redundant work per example, which dominates any evaluation
    over a real corpus.
    """
    cache_key = (_backend_name(backend) or "backend", candidate.synset_offset)
    cached = _SENSE_VECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    description = build_sense_description(candidate)
    result = backend.encode_text(description)
    if not result.is_valid or result.is_zero_vector:
        raise EmbeddingError(
            f"Sense encoding failed for {candidate.synset_id}: {result.error_message}"
        )

    _SENSE_VECTOR_CACHE[cache_key] = result.vector
    return result.vector


def clear_sense_vector_cache() -> None:
    """Drop cached sense vectors, for instance after switching embedding models."""
    _SENSE_VECTOR_CACHE.clear()


def _backend_name(embedding_backend: EmbeddingBackend | None) -> str | None:
    """Return the embedding model's name, or None when no backend is supplied."""
    if embedding_backend is None:
        return None
    info = embedding_backend.model_info
    return info.name if info else None


def _select_candidates(
    wn: WordNetService,
    sentence: str,
    target_word: str,
    occurrence_index: int,
    pos: str | None,
) -> tuple[list[CandidateSense], str | None]:
    """Choose candidate senses, narrowing by part of speech where possible.

    WordNet pools every part of speech for a lemma, so without this filter a
    noun sense of "mouse" competes against the verb "to prowl". An explicit POS
    from the caller wins; otherwise the target's tagged POS is used.

    Returns the candidates plus a note explaining the POS that was applied, or
    None when all parts of speech were considered.
    """
    if pos:
        scoped = wn.get_candidates(target_word, pos=pos)
        if scoped:
            return scoped, f"Candidates restricted to the supplied POS '{pos}'."
        # The requested POS has no senses for this word; fall through and warn
        # rather than returning nothing.
        unfiltered = wn.get_candidates(target_word)
        if unfiltered:
            return unfiltered, (
                f"No '{pos}' senses exist for '{target_word}'; all parts of speech were considered."
            )
        return [], None

    detected = tag_target_pos(sentence, target_word, occurrence_index)
    if detected:
        scoped = wn.get_candidates(target_word, pos=detected)
        if scoped:
            return scoped, f"Candidates restricted to the detected POS '{detected}'."

    unfiltered = wn.get_candidates(target_word)
    if detected:
        return unfiltered, (
            f"Detected POS '{detected}' yielded no senses; all parts of speech were considered."
        )
    return unfiltered, None


def disambiguate(
    sentence: str,
    target_word: str,
    occurrence_index: int,
    *,
    model_variant: ModelVariant,
    embedding_backend: EmbeddingBackend | None = None,
    context_strategy: ContextStrategy = ContextStrategy.LOCAL_WINDOW,
    context_window: int = 5,
    stopword_filtering: bool = False,
    lemmatize_context: bool = False,
    pos: str | None = None,
) -> PredictionResult:
    """Run WSD disambiguation using the specified model variant.

    Args:
        sentence: The input sentence.
        target_word: The word to disambiguate.
        occurrence_index: Which occurrence of target_word to use.
        model_variant: Which WSD method to use.
        embedding_backend: The embedding model to use. Optional, and ignored by
            the context-free baselines; required by the embedding variants.
        context_strategy: How to extract context.
        context_window: Window size for local context.
        stopword_filtering: Whether to filter stopwords from context.
        lemmatize_context: Whether to lemmatize context tokens.
        pos: Optional POS filter ('n', 'v', 'a', 'r'). When omitted, the target
            word's POS is detected from the sentence.

    Returns:
        A PredictionResult with the disambiguation outcome.
    """
    if model_variant.requires_embedding and embedding_backend is None:
        raise EmbeddingError(
            f"Model variant '{model_variant.display_name}' requires an embedding backend."
        )

    context_strategy = ContextStrategy(context_strategy)
    model_variant = ModelVariant(model_variant)

    wn = WordNetService()

    candidates, pos_note = _select_candidates(wn, sentence, target_word, occurrence_index, pos)

    if not candidates:
        return _failure_result(
            sentence,
            target_word,
            occurrence_index,
            context_strategy,
            model_variant,
            embedding_backend,
            [],
            error_message=f"No WordNet senses found for '{target_word}'.",
        )

    if model_variant == ModelVariant.FIRST_SENSE:
        return first_sense_baseline(
            sentence=sentence,
            target_word=target_word,
            occurrence_index=occurrence_index,
            context_strategy=context_strategy,
            embedding_model_name=_backend_name(embedding_backend),
            candidates=candidates,
            note=pos_note,
        )

    if model_variant == ModelVariant.MOST_FREQUENT_SENSE:
        return most_frequent_sense_baseline(
            sentence=sentence,
            target_word=target_word,
            occurrence_index=occurrence_index,
            context_strategy=context_strategy,
            embedding_model_name=_backend_name(embedding_backend),
            candidates=candidates,
            note=pos_note,
        )

    assert embedding_backend is not None  # guaranteed by requires_embedding above

    # Embedding-based methods
    extractor = ContextExtractor(
        context_window=context_window,
        stopword_filtering=stopword_filtering,
        lemmatize=lemmatize_context,
    )
    try:
        context_tokens = extractor.extract(
            sentence, target_word, occurrence_index, strategy=context_strategy
        )
    except DisambiguationError as exc:
        return _failure_result(
            sentence,
            target_word,
            occurrence_index,
            context_strategy,
            model_variant,
            embedding_backend,
            candidates,
            error_message=str(exc),
        )

    if not context_tokens:
        return _failure_result(
            sentence,
            target_word,
            occurrence_index,
            context_strategy,
            model_variant,
            embedding_backend,
            candidates,
            error_message="Empty context after extraction.",
            warnings=["Context was empty; disambiguation may be unreliable."],
        )

    try:
        context_vector = _encode_context(
            embedding_backend, context_tokens, strategy=context_strategy
        )
    except EmbeddingError as exc:
        return _failure_result(
            sentence,
            target_word,
            occurrence_index,
            context_strategy,
            model_variant,
            embedding_backend,
            candidates,
            context_tokens,
            error_message=f"Context encoding failed: {exc}",
        )

    warnings: list[str] = []
    if pos_note:
        warnings.append(pos_note)
    scored: list[tuple[float, int, CandidateSense]] = []

    for index, candidate in enumerate(candidates):
        try:
            sense_vector = _encode_sense(embedding_backend, candidate)
            success, score, error = cosine_similarity_safe(context_vector, sense_vector)

            if not success:
                warnings.append(f"Similarity failed for {candidate.synset_id}: {error}")
                continue

            assert score is not None
            scored.append((score, index, candidate))
        except EmbeddingError as exc:
            warnings.append(f"Encoding failed for {candidate.synset_id}: {exc}")
            continue

    if not scored:
        return _failure_result(
            sentence,
            target_word,
            occurrence_index,
            context_strategy,
            model_variant,
            embedding_backend,
            candidates,
            context_tokens,
            error_message="Could not compute similarity for any candidate sense.",
            warnings=warnings,
        )

    # Rank only after scoring: WordNet's candidate order breaks score ties.
    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked_candidates = [
        candidate.to_ranked(rank, score)
        for rank, (score, _, candidate) in enumerate(scored, start=1)
    ]

    predicted = ranked_candidates[0]

    return PredictionResult(
        sentence=sentence,
        target_word=target_word,
        selected_occurrence=occurrence_index,
        context=context_tokens,
        context_strategy=context_strategy,
        model_variant=model_variant,
        embedding_model_name=_backend_name(embedding_backend),
        candidates=candidates,
        ranked_candidates=ranked_candidates,
        predicted_synset_id=predicted.synset_id,
        predicted_definition=predicted.definition,
        top_similarity_score=predicted.similarity_score,
        success=True,
        warnings=warnings,
        processing_metadata={"pos_note": pos_note} if pos_note else {},
    )


def _failure_result(
    sentence: str,
    target_word: str,
    occurrence_index: int,
    context_strategy: ContextStrategy,
    model_variant: ModelVariant,
    embedding_backend: EmbeddingBackend | None,
    candidates: list[CandidateSense],
    context_tokens: list[str] | None = None,
    *,
    error_message: str,
    warnings: list[str] | None = None,
) -> PredictionResult:
    """Build a failure PredictionResult."""
    return PredictionResult(
        sentence=sentence,
        target_word=target_word,
        selected_occurrence=occurrence_index,
        context=context_tokens or [],
        context_strategy=context_strategy,
        model_variant=model_variant,
        embedding_model_name=_backend_name(embedding_backend),
        candidates=candidates,
        ranked_candidates=[],
        predicted_synset_id="",
        predicted_definition="",
        top_similarity_score=None,
        success=False,
        error_message=error_message,
        warnings=warnings or [],
    )
