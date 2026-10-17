"""Dataset loading and validation for WSD evaluation."""

from __future__ import annotations

import contextlib
import json
from enum import StrEnum
from pathlib import Path
from typing import IO

import pandas as pd

from contextsense_ai.config import settings
from contextsense_ai.embeddings.base import EmbeddingBackend
from contextsense_ai.exceptions import EvaluationError
from contextsense_ai.schemas import (
    ContextStrategy,
    DatasetExample,
    ModelVariant,
    PredictionResult,
)
from contextsense_ai.wordnet_service import WordNetService


class DatasetSource(StrEnum):
    """Supported dataset sources."""

    SAMPLE = "sample"
    CSV = "csv"


REQUIRED_COLUMNS = frozenset({"sentence", "target_word", "gold_synset"})


def validate_dataset_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate that a DataFrame has the required columns."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    errors: list[str] = []
    if missing:
        errors.append(f"Missing required columns: {', '.join(sorted(missing))}")
    return (len(errors) == 0, errors)


def validate_gold_synset(synset_id: str) -> bool:
    """Check whether a gold synset ID is valid.

    Validates the ``lemma.pos.sense_number`` format and, when WordNet is
    available, that the sense actually exists and contains that lemma.
    """
    if not synset_id or not isinstance(synset_id, str):
        return False

    parts = synset_id.split(".")
    if len(parts) != 3:
        return False

    lemma, pos, sense_number = parts
    if not lemma or not pos or not sense_number:
        return False

    if pos not in ("n", "v", "a", "r", "s"):
        return False

    if not sense_number.isdigit():
        return False

    try:
        return WordNetService().synset_exists(synset_id)
    except Exception:
        # If WordNet is unavailable, accept the ID on format alone.
        return True


def load_sample_dataset() -> list[DatasetExample]:
    """Load the built-in sample demonstration dataset.

    This dataset contains examples for ambiguous words like bank, bat, mouse, spring.
    It is NOT a substitute for a real annotated benchmark.
    """
    sample_path = settings.data_dir / "sample_examples.json"
    if not sample_path.exists():
        return _create_default_sample_examples()

    try:
        with open(sample_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise EvaluationError(f"Failed to load sample dataset: {exc}") from exc

    examples: list[DatasetExample] = []
    for idx, item in enumerate(data.get("examples", [])):
        gold = str(item.get("gold_synset", ""))
        valid = bool(gold) and validate_gold_synset(gold)
        examples.append(
            DatasetExample(
                id=f"sample_{idx}",
                sentence=str(item.get("sentence", "")),
                target_word=str(item.get("target_word", "")),
                target_occurrence=int(item.get("target_occurrence", 0)),
                gold_synset=gold,
                pos=str(item.get("pos", "n")) if item.get("pos") else None,
                dataset_name="sample",
                is_valid=valid,
                validation_error=None if valid else f"Invalid synset ID: '{gold}'",
            )
        )

    return examples


def _create_default_sample_examples() -> list[DatasetExample]:
    """Create default sample examples for demonstration.

    Gold labels are verified against WordNet 3.0 definitions; see
    ``data/sample_examples.json`` for the annotated equivalents.
    """
    rows = [
        ("I deposited money in the bank.", "bank", "bank.n.02"),
        ("The fisherman sat near the bank of the river.", "bank", "bank.n.01"),
        ("The baseball player hit the ball with a bat.", "bat", "bat.n.05"),
        ("A bat flew out of the cave.", "bat", "bat.n.01"),
        ("I moved the cursor using the mouse.", "mouse", "mouse.n.04"),
        ("The mouse ran across the kitchen floor.", "mouse", "mouse.n.01"),
        ("Spring brings new life to the garden.", "spring", "spring.n.01"),
        ("The spring in the mattress broke.", "spring", "spring.n.02"),
    ]
    return [
        DatasetExample(
            id=f"sample_{idx}",
            sentence=sentence,
            target_word=word,
            target_occurrence=0,
            gold_synset=gold,
            pos="n",
            dataset_name="sample",
        )
        for idx, (sentence, word, gold) in enumerate(rows)
    ]


def load_csv_dataset(
    filepath: str | Path | IO[bytes],
    *,
    sentence_col: str = "sentence",
    target_col: str = "target_word",
    gold_col: str = "gold_synset",
    occurrence_col: str | None = None,
    pos_col: str | None = None,
    id_col: str | None = None,
) -> list[DatasetExample]:
    """Load a CSV dataset for evaluation.

    Required columns: sentence, target_word, gold_synset.
    Optional columns: target_occurrence, pos, id.

    Args:
        filepath: Path to the CSV file, or an already-open binary stream.
        sentence_col: Column name for sentences.
        target_col: Column name for target words.
        gold_col: Column name for gold synset IDs.
        occurrence_col: Optional column for occurrence indices.
        pos_col: Optional column for POS tags.
        id_col: Optional column for example IDs.

    Returns:
        List of validated DatasetExample objects.
    """
    if isinstance(filepath, (str, Path)):
        path = Path(filepath)
        if not path.exists():
            raise EvaluationError(f"Dataset file not found: {path}")
        source: str | Path | IO[bytes] = path
        dataset_name = path.name
    else:
        # An uploaded file has no filesystem location; read straight off the stream.
        source = filepath
        dataset_name = getattr(filepath, "name", "uploaded.csv")

    try:
        df = pd.read_csv(source)
    except Exception as exc:
        raise EvaluationError(f"Failed to read CSV file: {exc}") from exc

    is_valid, errors = validate_dataset_schema(df)
    if not is_valid:
        raise EvaluationError(f"Invalid dataset schema: {'; '.join(errors)}")

    examples: list[DatasetExample] = []
    for idx, row in df.iterrows():
        if pd.isna(row[sentence_col]) or not str(row[sentence_col]).strip():
            examples.append(
                DatasetExample(
                    id=f"row_{idx}",
                    sentence="",
                    target_word=str(row[target_col]) if not pd.isna(row[target_col]) else "",
                    target_occurrence=0,
                    gold_synset="",
                    is_valid=False,
                    validation_error="Empty sentence.",
                )
            )
            continue

        gold = str(row[gold_col]) if not pd.isna(row[gold_col]) else ""
        if not validate_gold_synset(gold):
            examples.append(
                DatasetExample(
                    id=f"row_{idx}",
                    sentence=str(row[sentence_col]),
                    target_word=str(row[target_col]) if not pd.isna(row[target_col]) else "",
                    target_occurrence=0,
                    gold_synset=gold,
                    is_valid=False,
                    validation_error=f"Invalid synset ID: '{gold}'",
                )
            )
            continue

        occurrence = 0
        if occurrence_col and occurrence_col in df.columns:
            val = row[occurrence_col]
            if not pd.isna(val):
                with contextlib.suppress(ValueError, TypeError):
                    occurrence = int(val)

        pos = None
        if pos_col and pos_col in df.columns:
            val = row[pos_col]
            if not pd.isna(val):
                pos = str(val)

        ex_id = f"row_{idx}"
        if id_col and id_col in df.columns:
            val = row[id_col]
            if not pd.isna(val):
                ex_id = str(val)

        examples.append(
            DatasetExample(
                id=ex_id,
                sentence=str(row[sentence_col]),
                target_word=str(row[target_col]) if not pd.isna(row[target_col]) else "",
                target_occurrence=occurrence,
                gold_synset=gold,
                pos=pos,
                dataset_name=dataset_name,
                is_valid=True,
            )
        )

    return examples


def evaluate_example(
    example: DatasetExample,
    model_variant: ModelVariant,
    embedding_backend: EmbeddingBackend | None = None,
    *,
    context_strategy: ContextStrategy = ContextStrategy.LOCAL_WINDOW,
    context_window: int = 5,
    stopword_filtering: bool = False,
    lemmatize_context: bool = False,
) -> PredictionResult:
    """Run disambiguation on a single example and return the prediction."""
    from contextsense_ai.evaluation.metrics import predict_example

    return predict_example(
        example,
        model_variant,
        embedding_backend,
        context_strategy=context_strategy,
        context_window=context_window,
        stopword_filtering=stopword_filtering,
        lemmatize_context=lemmatize_context,
    )
