# Data Directory

This directory contains datasets and downloaded resources for ContextSense AI.

## Structure

```
data/
├── README.md              # This file
├── sample_examples.json   # Built-in demonstration examples
├── raw/                   # Raw downloaded datasets (not committed)
├── processed/            # Processed datasets (not committed)
└── word2vec/             # Word2Vec model files (not committed)
```

## Demonstration Data

### sample_examples.json

A small dataset with 8 examples for ambiguous words (bank, bat, mouse, spring).
**This is for demonstration only and is not a benchmark.**

Schema:
```json
{
  "sentence": "I deposited money in the bank.",
  "target_word": "bank",
  "target_occurrence": 0,
  "gold_synset": "bank.n.01",
  "pos": "n"
}
```

## Adding Real Datasets

For proper evaluation, add a real WSD dataset like SemCor to `data/raw/`.

### CSV Format

User-uploaded CSV files should have these columns:
- `sentence` (required): The text containing the target word
- `target_word` (required): The ambiguous word
- `gold_synset` (required): The WordNet synset ID (e.g., "bank.n.01")
- `target_occurrence` (optional): Which occurrence (0-indexed)
- `pos` (optional): Part of speech

### SemCor

SemCor is the largest manually word sense annotated corpus.
It is available from the [WordNet download page](https://wordnet.princeton.edu/wordnet/download/current-version/).

Place SemCor files in `data/raw/semcor/` after downloading.

## Word2Vec Model

Download the Google News Word2Vec model manually:
```bash
mkdir -p data/word2vec
curl -o data/word2vec/GoogleNews-vectors-negative300.bin.gz \
  "https://storage.googleapis.com/word2vec-tutorial-models/GoogleNews-vectors-negative300.bin.gz"
```

This model is ~3.6GB compressed and requires significant RAM to load.

## License

Check individual dataset licenses. WordNet data is licensed under
the WordNet License. SemCor has specific terms of use.
