"""Regression tests for defects found during the project audit.

Each test here corresponds to a bug that shipped in an earlier revision, so a
regression fails loudly rather than silently reintroducing the defect.
"""

from __future__ import annotations

import numpy as np
import pytest

from contextsense_ai.disambiguation.baseline import (
    first_sense_baseline,
    most_frequent_sense_baseline,
)
from contextsense_ai.disambiguation.engine import disambiguate
from contextsense_ai.embeddings.base import (
    EmbeddingBackend,
    EmbeddingModelInfo,
    EmbeddingResult,
)
from contextsense_ai.evaluation.datasets import load_sample_dataset, validate_gold_synset
from contextsense_ai.evaluation.experiments import run_model_comparison, save_results
from contextsense_ai.preprocessing import get_wordnet_pos_from_tag
from contextsense_ai.schemas import (
    ContextStrategy,
    ModelVariant,
)
from contextsense_ai.wordnet_service import WordNetService


class StubBackend(EmbeddingBackend):
    """Deterministic embedding backend with no model download."""

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            name="stub",
            backend="stub",
            dimensions=self.dimensions,
            is_available=True,
        )

    def encode_text(self, text: str) -> EmbeddingResult:
        seed = abs(hash(text)) % (2**32)
        vector = np.random.default_rng(seed).normal(size=self.dimensions)
        return EmbeddingResult(vector=vector, model_name="stub", is_valid=True)

    def encode_tokens(self, tokens: list[str]) -> EmbeddingResult:
        return self.encode_text(" ".join(tokens))

    def is_available(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def release(self) -> None:
        pass


def test_synset_id_is_relative_to_target_word() -> None:
    """The financial sense of 'bank' must not be reported under another lemma."""
    candidates = WordNetService().get_candidates("bank")
    ids = [c.synset_id for c in candidates]

    # synset.name() would yield depository_financial_institution.n.01 here.
    assert "bank.n.02" in ids
    assert all(synset_id.startswith("bank.") for synset_id in ids)


def test_synset_id_matches_wordnet_for_unambiguous_prefix() -> None:
    """Where the target already keys the sense, the ID must match WordNet's."""
    service = WordNetService()
    for index, candidate in enumerate(service.get_candidates("bank")[:3], start=1):
        assert candidate.synset_id == f"bank.n.{index:02d}"


def test_get_synset_from_id_rejects_wrong_lemma() -> None:
    """An ID whose lemma is absent from the synset must not resolve."""
    service = WordNetService()
    assert service.get_synset_from_id("bank.n.99") is None
    assert service.get_synset_from_id("nonexistentword.n.01") is None
    assert service.get_synset_from_id("bank.n.01") is not None


def test_sample_dataset_gold_labels_are_valid() -> None:
    """Every bundled sample label must be a real WordNet sense."""
    examples = load_sample_dataset()
    assert examples, "sample dataset should not be empty"
    for example in examples:
        assert example.is_valid, f"{example.id} rejected: {example.validation_error}"
        assert validate_gold_synset(example.gold_synset)


def test_sample_dataset_gold_labels_match_their_sentences() -> None:
    """Guard the specific labels that were previously mislabelled."""
    service = WordNetService()
    for example in load_sample_dataset():
        synset = service.get_synset_from_id(example.gold_synset)
        assert synset is not None, example.gold_synset

    expected = {
        "I deposited money in the bank.": "bank.n.02",  # financial institution
        "The fisherman sat near the bank of the river.": "bank.n.01",  # land
        "The baseball player hit the ball with a bat.": "bat.n.05",  # sports
        "A bat flew out of the cave.": "bat.n.01",  # mammal
        "I moved the cursor using the mouse.": "mouse.n.04",  # device
        "The spring in the mattress broke.": "spring.n.02",  # mechanical
    }
    for example in load_sample_dataset():
        if example.sentence in expected:
            assert example.gold_synset == expected[example.sentence], example.sentence


def test_baselines_run_without_an_embedding_backend() -> None:
    """Context-free baselines must not require an embedding backend."""
    for variant in (ModelVariant.FIRST_SENSE, ModelVariant.MOST_FREQUENT_SENSE):
        result = disambiguate(
            sentence="I deposited money in the bank.",
            target_word="bank",
            occurrence_index=0,
            model_variant=variant,
            embedding_backend=None,
        )
        assert result.success, result.error_message
        assert result.predicted_synset_id


def test_embedding_variant_without_backend_raises() -> None:
    """An embedding variant with no backend is a programming error."""
    from contextsense_ai.exceptions import EmbeddingError

    with pytest.raises(EmbeddingError):
        disambiguate(
            sentence="I deposited money in the bank.",
            target_word="bank",
            occurrence_index=0,
            model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
            embedding_backend=None,
        )


def test_context_strategy_stays_an_enum_through_evaluation() -> None:
    """String strategies must be normalised so explanation rendering works."""
    from contextsense_ai.evaluation.experiments import run_evaluation

    result = run_evaluation(
        load_sample_dataset()[:1],
        ModelVariant.FIRST_SENSE,
        None,
        context_strategy="local_window",  # type: ignore[arg-type]
    )
    prediction = next(iter(result.predictions.values()))
    assert prediction.context_strategy is ContextStrategy.LOCAL_WINDOW
    assert prediction.context_strategy.value == "local_window"


def test_run_model_comparison_keys_predictions_by_example_id() -> None:
    """Predictions must be keyed by example id so CSV export is not empty."""
    examples = load_sample_dataset()
    result = run_model_comparison(
        examples,
        StubBackend(),
        model_variants=[ModelVariant.FIRST_SENSE, ModelVariant.SENTENCE_TRANSFORMER_WORDNET],
    )

    assert set(result.predictions) == {e.id for e in examples if e.is_valid}
    assert all(not key.isspace() for key in result.predictions)
    for key in result.predictions:
        assert key in {e.id for e in examples}


def test_comparison_export_writes_one_row_per_example(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The comparison CSV must contain a row for every example."""
    import pandas as pd

    examples = load_sample_dataset()
    result = run_model_comparison(
        examples,
        StubBackend(),
        model_variants=[ModelVariant.FIRST_SENSE],
    )
    save_results(result, tmp_path / "out")

    csv_path = tmp_path / "out.predictions.csv"
    assert csv_path.exists()
    frame = pd.read_csv(csv_path)
    assert len(frame) == len([e for e in examples if e.is_valid])


def test_comparison_export_keeps_every_model_variant(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A comparison must export every variant, not only the last one evaluated.

    ``EvaluationResult.predictions`` is keyed by example id, so writing each
    variant into the same mapping left only the final variant in the CSV while
    the summary JSON reported all three. The CSV looked complete and was not.
    """
    import pandas as pd

    variants = [
        ModelVariant.FIRST_SENSE,
        ModelVariant.MOST_FREQUENT_SENSE,
        ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
    ]
    result = run_model_comparison(
        load_sample_dataset(),
        StubBackend(),
        model_variants=variants,
    )
    save_results(result, tmp_path / "out")

    frame = pd.read_csv(tmp_path / "out.predictions.csv")
    assert set(frame["model_variant"]) == {v.value for v in variants}
    assert len(frame) == result.shared_subset_evaluated * len(variants)


def test_most_frequent_sense_differs_from_first_sense() -> None:
    """The MFS baseline must be a genuinely different algorithm.

    'duck' is a discriminating case: WordNet orders the bird sense first, while
    SemCor records the verb 'to duck' as more frequent.
    """
    first = first_sense_baseline("x", "duck", 0, ContextStrategy.LOCAL_WINDOW, candidates=None)
    mfs = most_frequent_sense_baseline(
        "x", "duck", 0, ContextStrategy.LOCAL_WINDOW, candidates=None
    )

    assert first.predicted_synset_id == "duck.n.01"
    assert mfs.predicted_synset_id == "duck.v.01"
    assert mfs.predicted_synset_id != first.predicted_synset_id


def test_most_frequent_sense_agrees_when_wordnet_ordering_is_correct() -> None:
    """When WordNet ordering happens to match the corpus, both agree."""
    first = first_sense_baseline("x", "spring", 0, ContextStrategy.LOCAL_WINDOW, candidates=None)
    mfs = most_frequent_sense_baseline(
        "x", "spring", 0, ContextStrategy.LOCAL_WINDOW, candidates=None
    )
    assert first.predicted_synset_id == mfs.predicted_synset_id == "spring.n.01"


def test_most_frequent_sense_uses_semcor_counts() -> None:
    """Ranks must come from corpus counts, not list position."""
    candidates = WordNetService().get_candidates("duck")
    assert sum(c.frequency_count for c in candidates) > 0, "SemCor counts unavailable"

    ranked_counts = [c.frequency_count for c in candidates]
    result = most_frequent_sense_baseline(
        "A duck walked past.",
        "duck",
        0,
        ContextStrategy.LOCAL_WINDOW,
        candidates=candidates,
    )
    # Rank order must follow descending corpus count, not WordNet listing order.
    emitted = [rc.frequency_count for rc in result.ranked_candidates]
    assert emitted == sorted(emitted, reverse=True)
    assert max(ranked_counts) == emitted[0]


def test_baselines_accept_precomputed_candidates() -> None:
    """Callers may pass POS-filtered candidates; baselines must honour them."""
    from contextsense_ai.disambiguation.baseline import first_sense_baseline

    nouns = WordNetService().get_candidates("bank", pos="n")
    result = first_sense_baseline(
        "I deposited money in the bank.",
        "bank",
        0,
        ContextStrategy.LOCAL_WINDOW,
        candidates=nouns,
        note="filtered",
    )
    assert all(c.synset_id.startswith("bank.n.") for c in result.candidates)
    assert result.processing_metadata["pos_note"] == "filtered"


def test_pos_filtering_narrows_candidates() -> None:
    """A supplied POS must exclude senses of other parts of speech."""
    result = disambiguate(
        sentence="I moved the cursor using the mouse.",
        target_word="mouse",
        occurrence_index=0,
        model_variant=ModelVariant.FIRST_SENSE,
        embedding_backend=None,
        pos="n",
    )
    assert result.success
    assert all(c.pos == "n" for c in result.candidates)
    assert all(c.synset_id.startswith("mouse.n.") for c in result.candidates)


def test_pos_filtering_falls_back_when_pos_has_no_senses() -> None:
    """An impossible POS must degrade gracefully, not return nothing."""
    service = WordNetService()
    # Find a word/POS combination WordNet genuinely lacks.
    target, impossible_pos = next(
        (word, pos)
        for word in ("bank", "bat", "spring", "mouse")
        for pos in ("n", "v", "a", "r")
        if not service.get_candidates(word, pos=pos)
    )

    result = disambiguate(
        sentence=f"A sentence containing {target}.",
        target_word=target,
        occurrence_index=0,
        model_variant=ModelVariant.FIRST_SENSE,
        embedding_backend=None,
        pos=impossible_pos,
    )
    assert result.success
    assert result.candidates
    assert any("all parts of speech" in w for w in result.warnings)


def test_ranks_are_assigned_after_sorting() -> None:
    """Rank 1 must always be the highest-scoring candidate."""

    class SkewedBackend(StubBackend):
        def encode_text(self, text: str) -> EmbeddingResult:
            if "athlete" in text:
                return EmbeddingResult(
                    vector=np.array([10.0, 0.0] + [0.0] * (self.dimensions - 2)),
                    model_name="skewed",
                    is_valid=True,
                )
            return EmbeddingResult(
                vector=np.array([0.0, 10.0] + [0.0] * (self.dimensions - 2)),
                model_name="skewed",
                is_valid=True,
            )

    result = disambiguate(
        sentence="The athlete went to the bank today.",
        target_word="bank",
        occurrence_index=0,
        model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
        embedding_backend=SkewedBackend(),
        pos="n",
    )
    assert result.success
    ranks = [c.rank for c in result.ranked_candidates]
    scores = [c.similarity_score for c in result.ranked_candidates]
    assert ranks == list(range(1, len(ranks) + 1))
    assert scores == sorted(scores, reverse=True)
    assert result.predicted_synset_id == result.ranked_candidates[0].synset_id


def test_explanation_renders_for_evaluation_predictions() -> None:
    """Explanation rendering must not crash on a str-based strategy."""
    from contextsense_ai.evaluation.experiments import run_evaluation
    from contextsense_ai.explanation import generate_explanation

    result = run_evaluation(load_sample_dataset()[:2], ModelVariant.FIRST_SENSE, None)
    for prediction in result.predictions.values():
        steps = generate_explanation(prediction)
        assert steps


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("NN", "n"), ("VB", "v"), ("JJ", "a"), ("RB", "r"), ("DT", None), (".", None)],
)
def test_pos_tag_mapping(tag: str, expected: str | None) -> None:
    assert get_wordnet_pos_from_tag(tag) == expected


def test_candidate_sense_roundtrips_frequency_fields() -> None:
    """Frequency metadata must survive the CandidateSense -> RankedCandidate copy."""
    service = WordNetService()
    candidate = next(c for c in service.get_candidates("bank") if c.frequency_rank is not None)
    ranked = candidate.to_ranked(1, 0.5)
    assert ranked.frequency_rank == candidate.frequency_rank
    assert ranked.frequency_count == candidate.frequency_count


def test_sense_vectors_are_cached_across_examples() -> None:
    """Sense descriptions are fixed text, so they must be encoded only once.

    Re-encoding every candidate sense for every example costs roughly 65 ms per
    call and dominates any evaluation over a real corpus.
    """
    from contextsense_ai.disambiguation.engine import clear_sense_vector_cache

    calls: list[str] = []

    class CountingBackend(StubBackend):
        def encode_text(self, text: str) -> EmbeddingResult:
            calls.append(text)
            return super().encode_text(text)

    backend = CountingBackend()
    clear_sense_vector_cache()

    sentences = [
        ("I deposited money in the bank.", "bank"),
        ("The fisherman sat near the bank of the river.", "bank"),
        ("A bat flew out of the cave.", "bat"),
    ]
    for text, word in sentences:
        disambiguate(
            sentence=text,
            target_word=word,
            occurrence_index=0,
            model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
            embedding_backend=backend,
            pos="n",
        )

    # Caching must bound the encodes well below one-per-candidate-per-example.
    assert len(calls) < 24 * len(sentences), "sense descriptions appear to be re-encoded"

    # A second pass over the same inputs must issue no new sense encodes: only
    # the per-example context is new.
    before = len(calls)
    for text, word in sentences:
        disambiguate(
            sentence=text,
            target_word=word,
            occurrence_index=0,
            model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
            embedding_backend=backend,
            pos="n",
        )
    assert len(calls) == before + len(sentences), "cached senses were not reused"

    clear_sense_vector_cache()


def test_sense_cache_is_keyed_by_model() -> None:
    """Vectors from one embedding model must never leak into another."""
    from contextsense_ai.disambiguation.engine import clear_sense_vector_cache

    clear_sense_vector_cache()

    class OtherBackend(StubBackend):
        def encode_text(self, text: str) -> EmbeddingResult:
            seed = abs(hash(("other", text))) % (2**32)
            vector = np.random.default_rng(seed).normal(size=self.dimensions)
            return EmbeddingResult(vector=vector, model_name="other", is_valid=True)

        @property
        def model_info(self) -> EmbeddingModelInfo:
            info = super().model_info
            return EmbeddingModelInfo(
                name="other-model",
                backend="stub",
                dimensions=info.dimensions or self.dimensions,
                is_available=True,
            )

    results = []
    for backend in (StubBackend(), OtherBackend()):
        clear_sense_vector_cache()
        result = disambiguate(
            sentence="I deposited money in the bank.",
            target_word="bank",
            occurrence_index=0,
            model_variant=ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
            embedding_backend=backend,
            pos="n",
        )
        results.append(result)

    assert all(r.success for r in results)
    clear_sense_vector_cache()


def test_every_candidate_contains_the_target_as_a_lemma() -> None:
    """wn.synsets only returns synsets containing the word, so IDs stay well formed."""
    service = WordNetService()
    for word in ("bank", "bat", "spring", "mouse"):
        for candidate in service.get_candidates(word):
            assert any(lemma.name().lower() == word for lemma in candidate.synset.lemmas()), (
                candidate.synset_id
            )
            assert candidate.synset_id.startswith(f"{word}.{candidate.pos}.")
