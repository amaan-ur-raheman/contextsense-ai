# ContextSense AI - Evaluation

## Dataset Acquisition

### SemCor (recommended)

SemCor is the standard manually annotated word sense disambiguation corpus: every
token in a Brown-corpus sentence carries a WordNet 3.0 sense assignment. It is
part of the WordNet project and is distributed under the WordNet License.

**Source:** [WordNet Download Page](https://wordnet.princeton.edu/wordnet/download/current-version/)

**Acquisition:**

```bash
uv run python -m contextsense_ai.datasets.semcor --output data/datasets/semcor_brown.csv
```

This downloads the Princeton WordNet 3.0 database and the SemCor CSVs (~70 MB),
caches them under `data/raw/`, and writes roughly 130,000 evaluation examples
across about 15,000 lemmas. Nothing large is committed to the repository.

**Sense label translation.** SemCor identifies a sense by a WordNet sense key
whose `wn_sense` field is a *global* synset number, not the per-lemma index used
in names like `bank.n.02`. `artist` has exactly one sense, globally numbered 18,
so using that number directly produces `artist.n.18`, which does not exist. The
converter resolves `(pos, lemma, wn_sense)` to a synset offset through the
official `index.sense`, then converts the offset into per-lemma numbering using
NLTK's canonical ordering. Verify generated labels against WordNet if you modify
this step; a plausible-looking mistake here silently mislabels the corpus.

**Filtering.** Rows whose gold sense is absent from WordNet for that lemma (a few
plural lemmas such as "eggs") are dropped, as are rows whose target lemma is not
a standalone token in the sentence. The latter matters because the engine locates
targets by token match: keeping those rows means one model variant silently loses
examples the others score, and the comparison stops being like-for-like.

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

## Measured Results

Measured on a 3,000-example random sample of the converted SemCor corpus
(local-window context, window 5, `all-MiniLM-L6-v2`). All variants scored the
same rows. These are the numbers in the committed
`artifacts/benchmark_3k.summary.json`; an earlier 5,000-example run was not
retained, so it is not reported here.

| Model | Accuracy | F1 (weighted) | F1 (macro) | Scored |
| --- | --- | --- | --- | --- |
| WordNet first-sense | 0.3097 | 0.3038 | 0.2357 | 3000 |
| Most-frequent-sense | **0.3280** | 0.3235 | **0.2470** | 3000 |
| Sentence Transformers + WordNet | **0.3317** | **0.3521** | 0.2132 | 3000 |

Reproduce with:

```bash
uv run python -m contextsense_ai.datasets.semcor --output data/datasets/semcor_brown.csv
uv run run-evaluation --csv data/datasets/semcor_brown.csv --model all
```

### Reading these numbers honestly

SemCor is an **all-words** corpus: every content word is annotated, including
function words and rare terms, rather than a curated set of deliberately
ambiguous words. Absolute scores are correspondingly low, and roughly 30% is the
level a strong contextual system typically reaches here. The gap to the 0.875
seen on the bundled 8-example sample is entirely explained by that difference in
difficulty, not by any change in the system.

The embedding method leads by 0.4 accuracy points over the frequency baseline,
which is within noise on 3,000 rows, and it leads clearly on weighted F1. It
**loses on macro F1** (0.2132 against 0.2357 and 0.2470). Macro-averaged scores
are low across the board because the corpus contains thousands of hapax
legomena: a sense annotated once contributes a full 0 or 1 to the macro average.
The method concentrates probability on high-frequency senses, so it trades
rare-sense recall for it. Any claim that this approach beats the baselines should
carry that trade-off with it.

Coverage is equal across variants to within one row, so these numbers are
comparable. Check `examples_evaluated` in the exported summary before comparing
any two runs; differing coverage silently invalidates the comparison.

## Limitations of Results

1. **Sample dataset is not a benchmark**: The built-in 8 examples are for
   demonstration only. Do not report results from them as benchmark performance.

2. **Sense inventory alignment**: Results depend on exact synset ID matching.
   Different datasets may use different sense inventories. SemCor is annotated
   against WordNet 3.0, which is the inventory NLTK ships, so sense IDs line up.

3. **Coverage differences**: If models skip different examples, comparing
   raw metrics alone can be misleading. Always check coverage. Datasets are
   filtered so every variant can predict every row, but a custom CSV may not be.

4. **Small datasets**: With few examples per sense, per-class metrics
   may be unstable.

5. **Context window effects**: Local window context may miss relevant
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
