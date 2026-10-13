# ContextSense AI

**Context-Aware Word Sense Disambiguation Using WordNet and Word Embeddings**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-9A348E.svg)](https://github.com/astral-sh/uv)
[![Streamlit](https://img.shields.io/badge/Streamlit-web%20UI-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational Word Sense Disambiguation (WSD) system that combines lexical knowledge from WordNet with embedding-based semantic similarity. Built for undergraduate NLP projects, classroom demonstrations, and technical presentations.

![Architecture Diagram](docs/architecture.md)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Downloading Resources](#downloading-resources)
- [Running the Application](#running-the-application)
- [Model Comparison](#model-comparison)
- [Dataset Evaluation](#dataset-evaluation)
- [Testing and Linting](#testing-and-linting)
- [Model Selection and Resource Requirements](#model-selection-and-resource-requirements)
- [Dataset Schema](#dataset-schema)
- [Evaluation Workflow](#evaluation-workflow)
- [Sample Inputs](#sample-inputs)
- [Troubleshooting](#troubleshooting)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [References](#references)
- [License](#license)

## Problem Statement

Natural language contains ambiguous words that have multiple meanings. For example:

| Word | Sentence | Intended Meaning |
|------|----------|------------------|
| bank | "I deposited money in the **bank**." | Financial institution |
| bank | "The fisherman sat near the **bank** of the river." | Land beside a river |
| bat | "The baseball player hit the ball with a **bat**." | Sports equipment |
| bat | "A **bat** flew out of the cave." | Flying mammal |
| mouse | "I moved the cursor using the **mouse**." | Computer pointing device |
| mouse | "The **mouse** ran across the kitchen floor." | Small animal |

Humans disambiguate effortlessly using context. Computers need explicit methods to do the same.

**Research question:** Does combining lexical knowledge from WordNet with distributional or contextual embeddings provide useful improvements over simple sense-selection baselines?

## Objectives

1. Build a complete, modular WSD system suitable for academic projects
2. Implement multiple WSD methods for fair comparison
3. Provide an interactive Streamlit interface for demonstration
4. Support reproducible evaluation on labelled datasets
5. Document the system thoroughly for teaching and viva preparation
6. Maintain high code quality and test coverage

## Features

### Interactive Playground
- Enter sentences with ambiguous words
- Select target word occurrences
- Choose from four WSD methods
- Configure context extraction and preprocessing
- View ranked candidate senses with definitions and examples
- Visualize similarity scores

### Model Comparison
- Run multiple methods on the same input
- Compare predicted synsets and definitions
- Analyze agreement between models
- Understand when different methods agree or disagree

### Dataset Evaluation
- Evaluate on built-in sample examples
- Upload custom CSV datasets
- Compute accuracy, precision, recall, F1
- View per-sense performance metrics
- Export prediction results

### Educational Resources
- Comprehensive documentation
- Architecture diagrams
- Algorithm pseudocode
- 50+ viva questions with answers
- Explanation of embeddings, WordNet, and evaluation metrics

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12+ |
| Package Manager | uv |
| Web Interface | Streamlit |
| Lexical Database | NLTK + WordNet |
| Embeddings | Sentence Transformers, Gensim (Word2Vec) |
| Numerical Computing | NumPy |
| Machine Learning | scikit-learn |
| Data Handling | Pandas |
| Visualization | Plotly, Streamlit charts |
| Testing | pytest |
| Linting | Ruff |

All dependencies are declared in `pyproject.toml`. No API keys or paid services required.

## Architecture

```
Input Sentence → Tokenization → Target Word Lookup
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              WordNet Service    Context Extractor    Embedding Model
              (Candidate         (Window / Full       (Sentence
               Senses)           Sentence)            Transformers /
                                                         Word2Vec)
                    │                   │                   │
                    ▼                   ▼                   ▼
              Candidate          Context Tokens      Context Vector
              Descriptions       (filtered)          + Sense Vectors
                                                            │
                                                            ▼
                                                  Cosine Similarity
                                                            │
                                                            ▼
                                                  Ranking → Prediction
```

See [docs/architecture.md](docs/architecture.md) for detailed module descriptions.

## Project Structure

```
contextsense-ai/
├── app.py                          # Streamlit web application
├── pyproject.toml                  # Project configuration and dependencies
├── uv.lock                         # Locked dependency versions
├── .python-version                 # Python version pin
├── .gitignore                      # Git ignore patterns
├── .env.example                    # Environment configuration example
├── README.md                       # This file
├── src/contextsense_ai/
│   ├── __init__.py                 # Package exports
│   ├── config.py                   # Application settings
│   ├── exceptions.py               # Exception hierarchy
│   ├── schemas.py                  # Data classes (PredictionResult, etc.)
│   ├── preprocessing.py            # Tokenization, normalization, lemmatization
│   ├── wordnet_service.py          # WordNet candidate retrieval
│   ├── context_extractor.py        # Context window extraction
│   ├── explanation.py              # Prediction explanation generation
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py                 # EmbeddingBackend abstract interface
│   │   ├── word2vec.py             # Word2Vec backend (Gensim)
│   │   └── sentence_transformer.py # Sentence Transformers backend
│   ├── disambiguation/
│   │   ├── __init__.py
│   │   ├── baseline.py             # First-sense and MFS baselines
│   │   ├── similarity.py           # Cosine similarity calculation
│   │   └── engine.py               # Main disambiguation orchestrator
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── datasets.py             # Dataset loading and validation
│   │   ├── metrics.py              # Evaluation metrics computation
│   │   └── experiments.py          # Experiment running and export
│   ├── scripts/
│   │   ├── download_resources.py   # Download NLTK resources
│   │   └── run_evaluation.py       # CLI evaluation runner
│   └── utils/
│       ├── __init__.py
│       ├── resource_manager.py     # Resource lifecycle management
│       └── logging_config.py       # Logging setup
├── tests/
│   ├── test_preprocessing.py       # Preprocessing tests
│   ├── test_wordnet_service.py     # WordNet tests
│   ├── test_context_extractor.py   # Context extraction tests
│   ├── test_similarity.py          # Similarity calculation tests
│   ├── test_disambiguation.py      # Disambiguation engine tests
│   ├── test_datasets.py            # Dataset loading tests
│   ├── test_metrics.py             # Metrics computation tests
│   └── test_regressions.py         # Regression tests for fixed defects
├── data/
│   ├── README.md                   # Data directory documentation
│   ├── sample_examples.json        # Built-in demonstration examples
│   ├── .gitkeep
│   └── processed/
│       └── .gitkeep
├── artifacts/
│   └── .gitkeep                    # Evaluation results storage
└── docs/
    ├── architecture.md             # System architecture documentation
    ├── algorithm.md                # Algorithm description and pseudocode
    ├── evaluation.md               # Evaluation methodology
    └── viva_questions.md           # 50+ Q&A for viva preparation
```

## Setup and Installation

### Prerequisites

- Python 3.12 or later
- `uv` package manager

### Install uv

```bash
# macOS (Homebrew)
brew install uv

# Or install directly
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Initialize the Project

```bash
# Navigate to the project directory
cd contextsense-ai

# Install dependencies (creates virtual environment and uv.lock)
uv sync

# Or with dev dependencies for testing
uv sync --extra dev
```

### Verify Installation

```bash
# Check Python version
uv run python --version

# Verify imports work
uv run python -c "from contextsense import settings; print('OK')"
```

## Downloading Resources

### NLTK Resources (Required)

NLTK WordNet data is required for candidate sense retrieval. These download automatically on first use, but you can pre-download them:

```bash
uv run download-resources
```

This downloads:
- `wordnet` - WordNet database, including the SemCor sense-frequency counts
- `omw-1.4` - Open Multilingual WordNet
- `punkt_tab` - Tokenizer data
- `averaged_perceptron_tagger_eng` - POS tagger, used to narrow candidate senses
  by the target word's part of speech (optional; without it all parts of speech
  are considered)

### Sentence Transformers Model (Auto-downloaded)

The default embedding backend uses Sentence Transformers. The model downloads automatically on first use (~80-150MB):

- Model: `all-MiniLM-L6-v2` (384 dimensions, lightweight)
- No manual download needed
- Downloads to local cache on first encoding

### Word2Vec Model (Optional, Manual Download)

The Word2Vec backend requires manual download of a large model (~3.6GB):

```bash
mkdir -p data/word2vec
curl -o data/word2vec/GoogleNews-vectors-negative300.bin.gz \
  "https://storage.googleapis.com/word2vec-tutorial-models/GoogleNews-vectors-negative300.bin.gz"
```

Then the system can use it from `data/word2vec/`. **This model is NOT downloaded automatically.**

## Running the Application

### Start the Streamlit Web Interface

```bash
uv run streamlit run app.py
```

This opens a browser window at `http://localhost:8501` with the full interface.

### Using the Playground

1. Enter a sentence (e.g., "I deposited money in the bank.")
2. Enter the target word (e.g., "bank")
3. If the word appears multiple times, select the occurrence
4. Choose a WSD method from the dropdown
5. Configure context extraction (window size, strategy)
6. Optionally adjust preprocessing (stopword filtering, lemmatization)
7. Click **Disambiguate**
8. View the predicted sense, ranked candidates, and explanation

### Example Demonstrations

The interface includes example buttons for:

- **Bank: financial** - "I deposited money in the bank."
- **Bank: river** - "The fisherman sat near the bank of the river."
- **Bat: sports** - "The baseball player hit the ball with a bat."
- **Bat: animal** - "A bat flew out of the cave."
- **Mouse: device** - "I moved the cursor using the mouse."
- **Mouse: animal** - "The mouse ran across the kitchen floor."
- **Spring: season** - "Spring brings new life to the garden."
- **Spring: mechanical** - "The spring in the mattress broke."

Click any example button to populate the inputs and run the disambiguation.

## Model Comparison

The Model Comparison page lets you run the same input through all available methods:

1. Enter a sentence and target word
2. Click **Compare Models**
3. View predictions from each method side by side
4. See which methods agree or disagree
5. Note: Agreement does not guarantee correctness

## Dataset Evaluation

### Using the Sample Dataset

1. Go to the **Dataset Evaluation** tab
2. Click **Run Evaluation on Sample**
3. View metrics for each model variant
4. Download predictions as CSV

**Note:** The sample dataset has only 8 examples and is for demonstration only.

### Uploading a Custom Dataset

1. Prepare a CSV with required columns: `sentence`, `target_word`, `gold_synset`
2. Upload via the file uploader
3. Review validation results (valid/invalid counts)
4. Click **Run Evaluation on Uploaded Dataset**
5. View metrics and download results

### CSV Format

```csv
sentence,target_word,gold_synset,target_occurrence
"I deposited money in the bank.",bank,bank.n.01,0
"The fisherman sat near the bank of the river.",bank,bank.n.03,0
```

Required columns: `sentence`, `target_word`, `gold_synset`
Optional columns: `target_occurrence`, `pos`, `id`

### Running Evaluation from Command Line

```bash
# Evaluate on sample dataset
uv run run-evaluation --sample

# Evaluate on custom CSV
uv run run-evaluation --csv data/my_dataset.csv

# Compare all models
uv run run-evaluation --sample --model all

# Custom context settings
uv run run-evaluation --sample --context-window 3
```

## Testing and Linting

### Run Tests

```bash
uv run pytest
```

### Run with verbose output

```bash
uv run pytest -v
```

### Run specific test files

```bash
uv run pytest tests/test_preprocessing.py
uv run pytest tests/test_similarity.py
```

### Run Linting

```bash
uv run ruff check .
```

### Run formatting check

```bash
uv run ruff format --check .
```

### Fix lint issues automatically

```bash
uv run ruff check --fix .
uv run ruff format .
```

## Model Selection and Resource Requirements

| Model | Backend | Dimensions | Download Size | RAM Usage | Auto-download? |
|-------|---------|------------|---------------|-----------|----------------|
| Sentence Transformers (default) | sentence_transformers | 384 | ~80-150MB | ~500MB-2GB | Yes |
| Word2Vec Google News | word2vec | 300 | ~3.6GB | ~8GB+ | No (manual) |
| First-sense baseline | N/A | N/A | 0 | Minimal | N/A |
| Most-frequent-sense baseline | N/A | N/A | 0 | Minimal | N/A |

**Recommendation:** Use Sentence Transformers (default) for best balance of quality and convenience.

## Dataset Schema

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `sentence` | string | Text containing the target word |
| `target_word` | string | The ambiguous word |
| `gold_synset` | string | WordNet synset ID (e.g., "bank.n.01") |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `target_occurrence` | int | Which occurrence (0-indexed) |
| `pos` | string | Part of speech hint (n, v, a, r) |
| `id` | string | Unique example identifier |

### Synset ID Format

WordNet synset IDs use the format: `lemma.pos.offset`

Examples:
- `bank.n.01` - financial institution
- `bank.n.03` - river bank
- `bat.n.01` - sports equipment
- `bat.n.02` - flying mammal

## Evaluation Workflow

1. **Prepare dataset**: Create CSV with required columns or use sample data
2. **Select model**: Choose which WSD method(s) to evaluate
3. **Run evaluation**: Click button or use CLI script
4. **Review metrics**: Check accuracy, precision, recall, F1
5. **Analyze per-class performance**: See which senses are challenging
6. **Export results**: Download CSV of per-example predictions

### Understanding the Metrics

- **Accuracy**: Overall fraction of correct predictions
- **Precision (macro)**: Average precision across all senses
- **Recall (macro)**: Average recall across all senses
- **F1 (macro)**: Harmonic mean of macro precision and recall
- **Per-class metrics**: Performance for each individual sense

### Skipped Examples

Examples may be skipped due to:
- Invalid data (missing required fields)
- Invalid gold synset ID
- Model failure during prediction
- Empty context after extraction

Skipped examples are reported separately and excluded from metric denominators.

## Sample Inputs

### For the Playground

Try these sentences with their target words:

| Target | Sentence |
|--------|----------|
| bank | I deposited money in the bank. |
| bank | The fisherman sat near the bank of the river. |
| bat | The baseball player hit the ball with a bat. |
| bat | A bat flew out of the cave. |
| mouse | I moved the cursor using the mouse. |
| mouse | The mouse ran across the kitchen floor. |
| spring | Spring brings new life to the garden. |
| spring | The spring in the mattress broke. |

### For Dataset Evaluation

Create a CSV like this:

```csv
sentence,target_word,gold_synset
"I deposited money in the bank.",bank,bank.n.01
"The fisherman sat near the bank of the river.",bank,bank.n.03
```

## Troubleshooting

### NLTK resources not found

```
# Run the download script
uv run download-resources

# Or download manually
python -m nltk.downloader wordnet omw-1.4 punkt_tab averaged_perceptron_tagger_eng
```

### Sentence Transformer model download fails

- Check internet connectivity
- Check disk space (need ~100-200MB)
- The model caches locally after first download
- Try a different model in configuration

### Word2Vec model not available

- Word2Vec requires manual download (~3.6GB)
- Download from: https://storage.googleapis.com/word2vec-tutorial-models/GoogleNews-vectors-negative300.bin.gz
- Place in `data/word2vec/GoogleNews-vectors-negative300.bin.gz`
- Or use Sentence Transformers backend instead

### Target word not found in sentence

- Check spelling and casing
- The system is case-insensitive but requires the word to appear
- Punctuation attached to the word is handled automatically
- If the word appears multiple times, select the correct occurrence

### All predictions failing

- Check that NLTK resources are downloaded
- Check that the embedding model is available
- Verify the target word has WordNet senses
- Try a different target word or sentence

### Out of memory

- The Sentence Transformers model uses ~500MB-2GB RAM
- Word2Vec requires ~8GB+ RAM
- Close other applications or use a machine with more RAM
- Consider using the baseline methods which don't need embeddings

### Slow performance

- First prediction after loading the model is slow (model initialization)
- Subsequent predictions are faster (model is cached)
- Larger context windows mean more tokens to encode
- Word2Vec is slower than Sentence Transformers for encoding

## Configuration

Create a `.env` file based on `.env.example` to customize settings:

```bash
cp .env.example .env
```

Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `DATA_DIR` | data | Main data directory |
| `NLTK_DATA_DIR` | ~/.nltk_data | NLTK data location |
| `DEFAULT_EMBEDDING_BACKEND` | sentence_transformer | Default embedding model |
| `DEFAULT_SENTENCE_TRANSFORMER_MODEL` | all-MiniLM-L6-v2 | ST model name |
| `DEFAULT_CONTEXT_WINDOW` | 5 | Default context window size |
| `RANDOM_SEED` | 42 | Random seed for reproducibility |

## Limitations

1. **Not a production system**: This is an educational demonstration
2. **Context window limitations**: Local window may miss relevant words
3. **Sense description quality**: WordNet definitions may not distinguish closely related senses
4. **Mean pooling**: Loses word order and treats all words equally
5. **No training**: Uses pretrained models, not task-specific training
6. **Small demo dataset**: Sample examples are not a benchmark
7. **SemCor requires manual download**: Not automatically acquired
8. **Similarity ≠ probability**: Scores indicate vector alignment, not prediction confidence

## Future Improvements

1. Dependency parsing for better context extraction
2. Fine-tuned transformer models for WSD
3. Dedicated sense embeddings
4. Ensemble of multiple methods
5. Evaluation on SemCor or SemEval benchmarks
6. POS tagging integration
7. Domain-specific models
8. multilingual WSD support

## References

1. Miller, G. A. (1995). WordNet: A lexical database for English. *Communications of the ACM*, 38(11), 39-41.
2. Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*.
3. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP 2019*.
4. Navigli, R. (2009). Word sense disambiguation: A survey. *ACM Computing Surveys*, 41(2), 1-69.
5. Princeton University. (2024). WordNet. https://wordnet.princeton.edu/

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- NLTK team for WordNet interface
- Hugging Face for Sentence Transformers
- Google for Word2Vec pretrained vectors
- Princeton University for WordNet
