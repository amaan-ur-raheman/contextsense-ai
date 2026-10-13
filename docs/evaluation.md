# ContextSense AI - Evaluation

## Dataset Acquisition

### SemCor (Recommended for Real Evaluation)

SemCor is the largest manually word sense annotated corpus, part of the
WordNet project. It contains ~352,000 words annotated with WordNet senses.

**Source:** [WordNet Download Page](https://wordnet.princeton.edu/wordnet/download/current-version/)

**License:** WordNet License (free for research and commercial use with attribution)

**Format:** SemCor is distributed as XML files with annotated sentences.

**Download process:**
1. Download the WordNet package which includes SemCor
2. Extract to `data/raw/semcor/`
3. Parse XML to extract (sentence, target_word, synset_id) tuples

**Note:** SemCor requires manual download as it's distributed with WordNet.
The automatic download script does not fetch SemCor.

### Alternative Datasets

- **SemEval WSD tasks**: Various evaluation datasets from SemEval competitions
- **OMSTI**: Open Multilingual Word Sense Induction dataset
- **Custom datasets**: Create your own CSV with the schema below

## Dataset Schema

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `sentence` | string | The text containing the target word |
| `target_word` | string | The ambiguous word to disambiguate |
| `gold_synset` | string | WordNet synset ID (e.g., "bank.n.01") |

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `target_occurrence` | int | Which occurrence (0-indexed), default 0 |
| `pos` | string | Part of speech hint (n, v, a, r) |
| `id` | string | Unique example identifier |

### Example

```csv
sentence,target_word,gold_synset,target_occurrence
"I deposited money in the bank.",bank,bank.n.01,0
"The fisherman sat near the bank of the river.",bank,bank.n.03,0
```

## Gold Labels

Gold labels must be valid WordNet synset IDs in the format `lemma.pos.offset`.

Example valid synset IDs:
- `bank.n.01` - financial institution
- `bank.n.03` - river bank
- `bat.n.01` - sports equipment
- `bat.n.02` - flying mammal

The evaluation validates that each gold synset ID exists in WordNet.

## Evaluation Metrics

### Accuracy

The fraction of correct predictions:
```
accuracy = correct_predictions / total_evaluated
```

### Precision (Macro)

Average precision across all senses, treating each sense equally:
```
precision_macro = mean(precision_i for each sense i)
```

### Recall (Macro)

Average recall across all senses:
```
recall_macro = mean(recall_i for each sense i)
```

### F1 Score

Harmonic mean of precision and recall:
```
f1 = 2 × (precision × recall) / (precision + recall)
```

We report both:
- **Macro F1**: Average across senses (treats rare senses equally)
- **Weighted F1**: Weighted by sense frequency

### Per-Class Metrics

For each sense, we compute:
- Precision: How often predictions of this sense are correct
- Recall: How often this sense is correctly predicted
- F1: Balance of precision and recall
- Support: Number of examples with this gold label

## Baselines

### First-Sense Baseline
Always predicts the first WordNet sense for the target word.

**Characteristics:**
- Does not use context
- Serves as a lower bound
- Performance depends on WordNet's sense ordering
- Should not be assumed to be the most frequent sense

### Most-Frequent-Sense Baseline
Predicts the sense with the highest corpus frequency.

**Characteristics:**
- Ranks senses by their SemCor (WordNet 3.0 sense-disambiguated corpus) counts
- A genuine frequency signal, so it differs from the first-sense baseline
- Senses with a corpus count of 0 are treated as unattested and ranked last,
  rather than being scored as frequency zero
- Falls back to WordNet ordering, and says so in a warning, when no sense of the
  word is attested in SemCor
- Does not use the sentence context

**Example.** For *bank*, WordNet lists the river-bank sense first while SemCor
records the financial sense as more frequent, so the two baselines disagree.

## Shared Test Subsets

When comparing multiple model variants, we evaluate on the same eligible examples:

1. Identify examples valid for all compared models
2. Report metrics on this shared subset
3. Also report per-model coverage (examples each model could evaluate)

This ensures fair comparison when different models have different
capabilities (e.g., embedding-based vs. baseline methods).

## Skipped Examples

Examples may be skipped for various reasons:

- **Invalid data**: Missing sentence, target word, or gold synset
- **Invalid gold label**: Gold synset ID not found in WordNet
- **Model failure**: Disambiguation returned an error for this example
- **Empty context**: Context extraction produced no tokens
- **Embedding failure**: Context or sense encoding failed

Skipped examples are reported separately from evaluation metrics.
They are NOT included in the denominator when computing accuracy.

## Reproducibility

To ensure reproducible evaluation:

1. **Fixed random seed**: Set `RANDOM_SEED=42` in settings
2. **Version control**: Record model names and dataset versions
3. **Configuration logging**: Save complete evaluation config with results
4. **Per-example predictions**: Export detailed predictions for inspection

### Saved Artifacts

Evaluation results include:

- `*.predictions.csv`: Per-example predictions with gold and predicted synsets
- `*.summary.json`: Aggregate metrics and configuration

## Limitations of Results

1. **Sample dataset is not a benchmark**: The built-in 8 examples are for
   demonstration only. Do not report results from them as benchmark performance.

2. **SemCor licensing**: SemCor requires manual download and has specific
   terms of use. Automatic acquisition may not be possible.

3. **Sense inventory alignment**: Results depend on exact synset ID matching.
   Different datasets may use different sense inventories.

4. **Coverage differences**: If models skip different examples, comparing
   raw metrics alone can be misleading. Always check coverage.

5. **Small datasets**: With few examples per sense, per-class metrics
   may be unstable.

6. **Context window effects**: Local window context may miss relevant
   words outside the window. Full-sentence context may include noise.

## Running Evaluation

```bash
# Using the sample dataset (for demonstration)
uv run run-evaluation --sample

# Using a custom CSV
uv run run-evaluation --csv path/to/dataset.csv

# Comparing multiple models
uv run run-evaluation --sample --model all

# With custom context settings
uv run run-evaluation --sample --context-window 3
```
