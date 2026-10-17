"""Typed data structures for predictions, candidates, and evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nltk.corpus import Synset


class ModelVariant(StrEnum):
    """Supported WSD model variants."""

    FIRST_SENSE = "first_sense"
    MOST_FREQUENT_SENSE = "most_frequent_sense"
    WORD2VEC_WORDNET = "word2vec_wordnet"
    SENTENCE_TRANSFORMER_WORDNET = "sentence_transformer_wordnet"

    @property
    def display_name(self) -> str:
        """Human-readable model name."""
        names = {
            ModelVariant.FIRST_SENSE: "WordNet First-Sense Baseline",
            ModelVariant.MOST_FREQUENT_SENSE: "Most-Frequent-Sense Baseline",
            ModelVariant.WORD2VEC_WORDNET: "Word2Vec + WordNet",
            ModelVariant.SENTENCE_TRANSFORMER_WORDNET: "Sentence Transformers + WordNet",
        }
        return names[self]

    @property
    def requires_embedding(self) -> bool:
        """Whether this variant requires an embedding model."""
        return self in (
            ModelVariant.WORD2VEC_WORDNET,
            ModelVariant.SENTENCE_TRANSFORMER_WORDNET,
        )


class ContextStrategy(StrEnum):
    """Context extraction strategies."""

    FULL_SENTENCE = "full_sentence"
    LOCAL_WINDOW = "local_window"


@dataclass
class CandidateSense:
    """A single candidate sense from WordNet."""

    synset: Synset
    pos: str
    lemmas: list[str]
    definition: str
    examples: list[str]
    hypernyms: list[str]
    hyponyms: list[str]
    synset_id: str = ""
    frequency_rank: int | None = None
    frequency_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.synset_id:
            self.synset_id = self.synset.name()

    @property
    def synset_offset(self) -> str:
        """Return the synset offset ID for evaluation matching."""
        return str(self.synset.offset())

    def to_ranked(self, rank: int, score: float) -> RankedCandidate:
        """Convert to a ranked candidate with a similarity score."""
        return RankedCandidate(
            rank=rank,
            synset_id=self.synset_id,
            pos=self.pos,
            lemmas=self.lemmas,
            definition=self.definition,
            examples=self.examples[:3],
            hypernyms=self.hypernyms[:5],
            hyponyms=self.hyponyms[:5],
            similarity_score=score,
            frequency_rank=self.frequency_rank,
            frequency_count=self.frequency_count,
            warnings=self.warnings.copy(),
        )


@dataclass
class RankedCandidate:
    """A ranked candidate sense with its similarity score."""

    rank: int
    synset_id: str
    pos: str
    lemmas: list[str]
    definition: str
    examples: list[str]
    hypernyms: list[str]
    hyponyms: list[str]
    similarity_score: float
    frequency_rank: int | None = None
    frequency_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_top_rank(self) -> bool:
        """Whether this candidate is ranked first."""
        return self.rank == 1


@dataclass
class PredictionResult:
    """Structured result of a disambiguation prediction."""

    sentence: str
    target_word: str
    selected_occurrence: int
    context: list[str]
    context_strategy: ContextStrategy
    model_variant: ModelVariant
    embedding_model_name: str | None
    candidates: list[CandidateSense]
    ranked_candidates: list[RankedCandidate]
    predicted_synset_id: str
    predicted_definition: str
    top_similarity_score: float | None
    success: bool
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)
    processing_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def predicted_synset(self) -> str:
        """Return the predicted synset identifier."""
        return self.predicted_synset_id

    def get_explanation_steps(self) -> list[tuple[str, str]]:
        """Return explanation steps for the prediction."""
        steps: list[tuple[str, str]] = []

        if not self.success:
            steps.append(("Error", self.error_message or "Unknown error occurred."))
            return steps

        steps.append(
            (
                "Candidate senses",
                f"Found {len(self.candidates)} possible meaning(s) for "
                f"'{self.target_word}' in WordNet.",
            )
        )

        steps.append(
            (
                "Context used",
                f"Strategy: {self.context_strategy.value}. "
                f"Context tokens: {', '.join(self.context[:10])}"
                f"{'...' if len(self.context) > 10 else ''}.",
            )
        )

        if self.embedding_model_name:
            steps.append(
                (
                    "Representation",
                    f"Context and sense descriptions encoded using {self.embedding_model_name}.",
                )
            )

        steps.append(
            (
                "Similarity",
                "Cosine similarity computed between context vector and "
                "each candidate sense vector.",
            )
        )

        if self.ranked_candidates:
            top = self.ranked_candidates[0]
            steps.append(
                (
                    "Selected sense",
                    f"Rank 1: {top.synset_id} - {top.definition[:80]}... "
                    f"(score: {top.similarity_score:.4f})",
                )
            )

        for warning in self.warnings:
            steps.append(("Warning", warning))

        return steps


@dataclass
class DatasetExample:
    """A single labelled example for evaluation."""

    id: str
    sentence: str
    target_word: str
    target_occurrence: int
    gold_synset: str
    pos: str | None = None
    dataset_name: str | None = None
    is_valid: bool = True
    validation_error: str | None = None


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single model variant."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    examples_evaluated: int
    examples_skipped: int
    examples_invalid: int
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] | None = None
    labels: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Complete evaluation results for one or more model variants."""

    dataset_name: str
    dataset_source: str
    examples: list[DatasetExample]
    per_model_results: dict[ModelVariant, EvaluationMetrics]
    shared_subset_evaluated: int
    config: dict[str, object]
    timestamp: str = ""
    predictions: dict[str, PredictionResult] = field(default_factory=dict)
    predictions_by_variant: dict[str, dict[str, PredictionResult]] = field(default_factory=dict)
