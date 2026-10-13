# Evaluation Datasets

## What is here

The bundled `sample_examples.json` contains **8 hand-written examples**. It exists
to demonstrate the interface. It is far too small to measure anything: a single
example is worth 12.5 accuracy points, and the words involved were chosen because
they are famously ambiguous.

Real numbers come from **SemCor**, fetched on demand.

## SemCor (recommended)

SemCor is the standard manually annotated word sense disambiguation corpus: every
token in a Brown-corpus sentence carries a WordNet 3.0 sense assignment.

```bash
uv run python -m contextsense_ai.datasets.semcor --output data/datasets/semcor_brown.csv
```

This downloads roughly 70 MB (Princeton WordNet 3.0 plus the SemCor CSVs),
converts them, and writes the result. Downloads are cached under `data/raw/`, so
subsequent runs are fast. Nothing large is committed to the repository.

Output columns match the CSV loader:

| Column | Meaning |
| --- | --- |
| `sentence` | The full sentence |
| `target_word` | The disambiguated lemma (not the inflected surface form) |
| `gold_synset` | Gold sense as `lemma.pos.NN` |
| `target_occurrence` | Which occurrence of the lemma to disambiguate (0-based) |
| `pos` | Part of speech |

Then evaluate:

```bash
uv run run-evaluation --csv data/datasets/semcor_brown.csv --model all
```

### Why conversion is non-trivial

SemCor identifies a sense by a WordNet **sense key**, and its `wn_sense` field is
a *global* synset number, not the per-lemma index used in names like `bank.n.02`.
`artist` has exactly one sense, globally numbered 18 — so naively using that
number produces `artist.n.18`, which does not exist.

The converter instead resolves `(pos, lemma, wn_sense)` to a synset offset via the
official WordNet 3.0 `index.sense`, then translates the offset into per-lemma
numbering using NLTK's own canonical ordering. Getting this wrong silently
mislabels the corpus, so it is worth checking that generated labels resolve
against WordNet.

### Filtering

Two kinds of row are dropped:

- rows whose gold sense does not exist in WordNet for that lemma (a few plural
  lemmas such as "eggs" and "woods"), which no prediction could ever match;
- rows whose target lemma is not a standalone token in the reconstructed
  sentence, because the engine locates targets by token match and would fail on
  them at prediction time.

The second matters more than it looks: without it, one model variant silently
loses rows the others score, and the comparison stops being like-for-like.

## Other datasets

Any CSV with `sentence`, `target_word`, and `gold_synset` columns works:

```bash
uv run run-evaluation --csv path/to/dataset.csv --model all
```

Optional columns: `target_occurrence`, `pos`, `id`. Gold labels are validated
against WordNet on load, and invalid rows are reported rather than scored.

See `docs/evaluation.md` for methodology and metric definitions.
