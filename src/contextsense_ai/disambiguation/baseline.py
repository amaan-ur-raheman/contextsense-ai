"""Baseline WSD methods: first-sense and most-frequent-sense.

Both baselines deliberately ignore the sentence context. They exist to give the
embedding-based methods something honest to beat.
"""

from __future__ import annotations

from contextsense_ai.schemas import (
    CandidateSense,
    ContextStrategy,
    ModelVariant,
    PredictionResult,
    RankedCandidate,
)
from contextsense_ai.wordnet_service import WordNetService

_NOT_PRESENT = 0.0


def _baseline_result(
    *,
    sentence: str,
    target_word: str,
    occurrence_index: int,
    context_strategy: ContextStrategy,
    model_variant: ModelVariant,
    embedding_model_name: str | None,
    candidates: list[CandidateSense] | None,
    note: str | None,
) -> PredictionResult:
    """Shared shell for the context-free baselines.

    ``candidates`` is normally supplied by the caller, which has already applied
    any part-of-speech filtering; it is looked up here only as a fallback.

    Returns a ``success=False`` result when the word has no WordNet senses, and
    otherwise a result with ``candidates`` populated but no prediction yet.
    """
    if candidates is None:
        candidates = WordNetService().get_candidates(target_word)

    result = PredictionResult(
        sentence=sentence,
        target_word=target_word,
        selected_occurrence=occurrence_index,
        context=[],
        context_strategy=context_strategy,
        model_variant=model_variant,
        embedding_model_name=embedding_model_name,
        candidates=candidates,
        ranked_candidates=[],
        predicted_synset_id="",
        predicted_definition="",
        top_similarity_score=None,
        success=False,
    )

    if not candidates:
        result.error_message = f"No WordNet senses found for '{target_word}'."
        result.warnings = ["This is a baseline method that does not use context."]

    result.processing_metadata = {"pos_note": note} if note else {}

    return result


def _finalize(
    result: PredictionResult,
    ranked: list[RankedCandidate],
    warnings: list[str],
) -> PredictionResult:
    """Attach a ranked candidate list and mark the result successful."""
    note = result.processing_metadata.get("pos_note")
    if isinstance(note, str) and note:
        warnings = [note, *warnings]

    result.ranked_candidates = ranked
    result.predicted_synset_id = ranked[0].synset_id
    result.predicted_definition = ranked[0].definition
    result.top_similarity_score = ranked[0].similarity_score
    result.success = True
    result.warnings = warnings
    return result


def first_sense_baseline(
    sentence: str,
    target_word: str,
    occurrence_index: int,
    context_strategy: ContextStrategy,
    *,
    embedding_model_name: str | None = None,
    candidates: list[CandidateSense] | None = None,
    note: str | None = None,
    **_context_kwargs: object,
) -> PredictionResult:
    """First-sense baseline: always picks WordNet's first sense.

    WordNet orders senses so that the first is intended to be the most common
    use, but that ordering is editorial rather than corpus-derived.
    """
    result = _baseline_result(
        sentence=sentence,
        target_word=target_word,
        occurrence_index=occurrence_index,
        context_strategy=context_strategy,
        model_variant=ModelVariant.FIRST_SENSE,
        embedding_model_name=embedding_model_name,
        candidates=candidates,
        note=note,
    )
    if not result.candidates:
        return result

    ranked = [
        cand.to_ranked(rank, 1.0 if rank == 1 else _NOT_PRESENT)
        for rank, cand in enumerate(result.candidates, start=1)
    ]

    return _finalize(
        result,
        ranked,
        warnings=[
            "First-sense baseline: ignores context and always returns WordNet's first sense.",
            "Similarity scores are synthetic (1.0 for rank 1, 0.0 otherwise), not computed.",
        ],
    )


def most_frequent_sense_baseline(
    sentence: str,
    target_word: str,
    occurrence_index: int,
    context_strategy: ContextStrategy,
    *,
    embedding_model_name: str | None = None,
    candidates: list[CandidateSense] | None = None,
    note: str | None = None,
    **_context_kwargs: object,
) -> PredictionResult:
    """Most-frequent-sense baseline: ranks senses by SemCor corpus frequency.

    Frequencies come from the WordNet 3.0 sense-disambiguated corpus, where each
    lemma carries a per-sense occurrence count. This is a genuine corpus signal,
    not a restatement of WordNet's editorial sense ordering.

    Senses the corpus never attests have a count of 0. SemCor is small, so a
    zero is far more likely to mean "not attested" than "genuinely rare"; those
    senses are ranked below every attested sense rather than being discarded.
    """
    result = _baseline_result(
        sentence=sentence,
        target_word=target_word,
        occurrence_index=occurrence_index,
        context_strategy=context_strategy,
        model_variant=ModelVariant.MOST_FREQUENT_SENSE,
        embedding_model_name=embedding_model_name,
        candidates=candidates,
        note=note,
    )
    if not result.candidates:
        return result

    senses: list[CandidateSense] = result.candidates
    total = sum(c.frequency_count for c in senses)

    if total == 0:
        warnings = [
            "Most-frequent-sense baseline: ignores context and ranks senses by SemCor frequency.",
            "No sense of this word is attested in SemCor; fell back to WordNet's own ordering, "
            "which makes this baseline identical to the first-sense baseline for this word.",
        ]
        ordered = list(senses)
    else:
        warnings = [
            "Most-frequent-sense baseline: ignores context and ranks senses by SemCor frequency.",
        ]
        unattested = sum(1 for c in senses if c.frequency_count == 0)
        if unattested:
            warnings.append(
                f"{unattested} of {len(senses)} senses are unattested in SemCor "
                "and were ranked below all attested senses."
            )
        # Stable sort keeps WordNet's ordering as the tie-break.
        ordered = sorted(senses, key=lambda c: -c.frequency_count)

    ranked = [
        cand.to_ranked(rank, cand.frequency_count / total if total else 0.0)
        for rank, cand in enumerate(ordered, start=1)
    ]

    return _finalize(result, ranked, warnings)
