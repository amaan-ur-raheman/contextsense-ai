"""Tests for dataset loading module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from contextsense_ai.evaluation.datasets import (
    load_csv_dataset,
    load_sample_dataset,
    validate_dataset_schema,
    validate_gold_synset,
)
from contextsense_ai.exceptions import EvaluationError
from contextsense_ai.schemas import DatasetExample


class TestLoadSampleDataset:
    """Tests for loading the sample dataset."""

    def test_load_sample_dataset(self) -> None:
        """Test loading the built-in sample dataset."""
        examples = load_sample_dataset()
        assert len(examples) > 0
        assert all(isinstance(e, DatasetExample) for e in examples)

    def test_sample_dataset_has_bank_examples(self) -> None:
        """Test that sample dataset contains bank examples."""
        examples = load_sample_dataset()
        bank_examples = [e for e in examples if e.target_word == "bank"]
        assert len(bank_examples) > 0

    def test_sample_dataset_examples_are_valid(self) -> None:
        """Test that sample dataset examples are valid."""
        examples = load_sample_dataset()
        for example in examples:
            assert example.is_valid is True
            assert example.gold_synset  # Has a gold synset

    def test_sample_dataset_sentences(self) -> None:
        """Test that sample dataset has expected sentences."""
        examples = load_sample_dataset()
        sentences = [e.sentence for e in examples]
        assert any("bank" in s.lower() for s in sentences)
        assert any("bat" in s.lower() for s in sentences)


class TestValidateDatasetSchema:
    """Tests for dataset schema validation."""

    def test_valid_schema(self) -> None:
        """Test validation of a valid schema."""
        df = pd.DataFrame(
            {
                "sentence": ["test"],
                "target_word": ["test"],
                "gold_synset": ["test.n.01"],
            }
        )
        is_valid, errors = validate_dataset_schema(df)
        assert is_valid is True
        assert len(errors) == 0

    def test_missing_sentence_column(self) -> None:
        """Test validation fails when sentence column is missing."""
        df = pd.DataFrame(
            {
                "target_word": ["test"],
                "gold_synset": ["test.n.01"],
            }
        )
        is_valid, errors = validate_dataset_schema(df)
        assert is_valid is False
        assert any("sentence" in e.lower() for e in errors)

    def test_missing_target_column(self) -> None:
        """Test validation fails when target_word column is missing."""
        df = pd.DataFrame(
            {
                "sentence": ["test"],
                "gold_synset": ["test.n.01"],
            }
        )
        is_valid, errors = validate_dataset_schema(df)
        assert is_valid is False

    def test_missing_gold_column(self) -> None:
        """Test validation fails when gold_synset column is missing."""
        df = pd.DataFrame(
            {
                "sentence": ["test"],
                "target_word": ["test"],
            }
        )
        is_valid, errors = validate_dataset_schema(df)
        assert is_valid is False


class TestLoadCsvDataset:
    """Tests for loading CSV datasets."""

    def test_load_valid_csv(self, tmp_path: Path) -> None:
        """Test loading a valid CSV dataset."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "sentence,target_word,gold_synset\n"
            "The bank approved the loan.,bank,bank.n.01\n"
            "The fisherman sat by the river bank.,bank,bank.n.03\n"
        )

        examples = load_csv_dataset(csv_path)
        assert len(examples) == 2
        assert all(e.is_valid for e in examples)

    def test_load_csv_with_optional_columns(self, tmp_path: Path) -> None:
        """Test loading CSV with optional columns."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "sentence,target_word,gold_synset,target_occurrence,pos\n"
            "The bank approved the loan.,bank,bank.n.01,0,n\n"
        )

        examples = load_csv_dataset(
            csv_path,
            occurrence_col="target_occurrence",
            pos_col="pos",
        )
        assert len(examples) == 1
        assert examples[0].target_occurrence == 0
        assert examples[0].pos == "n"

    def test_load_csv_invalid_synset(self, tmp_path: Path) -> None:
        """Test loading CSV with invalid synset IDs."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("sentence,target_word,gold_synset\ntest sentence,bank,invalid.synset\n")

        examples = load_csv_dataset(csv_path)
        assert len(examples) == 1
        assert examples[0].is_valid is False

    def test_load_csv_empty_sentence(self, tmp_path: Path) -> None:
        """Test loading CSV with empty sentences."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("sentence,target_word,gold_synset\n,bank,bank.n.01\n")

        examples = load_csv_dataset(csv_path)
        assert len(examples) == 1
        assert examples[0].is_valid is False

    def test_load_csv_file_not_found(self, tmp_path: Path) -> None:
        """Test error when CSV file doesn't exist."""
        csv_path = tmp_path / "nonexistent.csv"
        with pytest.raises(EvaluationError):
            load_csv_dataset(csv_path)

    def test_load_csv_invalid_schema(self, tmp_path: Path) -> None:
        """Test error when CSV has invalid schema."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("wrong_column,another_column\nvalue1,value2\n")

        with pytest.raises(EvaluationError):
            load_csv_dataset(csv_path)


class TestValidateGoldSynset:
    """Tests for gold synset validation."""

    def test_valid_synset_ids(self) -> None:
        """Test validation of valid synset IDs."""
        assert validate_gold_synset("bank.n.01") is True
        assert validate_gold_synset("bat.n.01") is True

    def test_invalid_synset_ids(self) -> None:
        """Test validation of invalid synset IDs."""
        assert validate_gold_synset("") is False
        assert validate_gold_synset("invalid.x.999999") is False
        assert validate_gold_synset("not.a.real.synset") is False

    def test_none_synset(self) -> None:
        """Test validation of None."""
        assert validate_gold_synset(None) is False  # type: ignore
