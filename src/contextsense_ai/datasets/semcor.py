"""Dataset acquisition and conversion for real WSD benchmarks.

The bundled 8-example sample set is for demonstration only. This module fetches
SemCor, the standard manually annotated WSD corpus, and converts it into the
project's evaluation schema.

SemCor is distributed with the Princeton WordNet package rather than as a
standalone download, so the flow is:

1. download the WordNet 3.0 database (needed for ``index.sense``);
2. download the SemCor CSVs, which mirror the Brown corpus annotation;
3. convert the token-level annotation into one row per disambiguated word.

Usage::

    uv run python -m contextsense_ai.datasets.semcor --output data/datasets/semcor.csv

Gold labels are only trustworthy if the sense numbering is translated correctly;
see :func:`load_sense_to_offset` and :func:`build_offset_to_sense_id`.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from contextsense_ai.utils.resource_manager import ensure_nltk_resources
from contextsense_ai.wordnet_service import WordNetService

#: Princeton WordNet 3.0 database. Contains ``index.sense``, the authoritative
#: mapping from a lemma's WordNet sense key to a synset offset.
WORDNET3_URL = "https://wordnetcode.princeton.edu/3.0/WNdb-3.0.tar.gz"

#: SemCor distributed as CSV, one file per Brown corpus section.
SEMCOR_BASE = "https://huggingface.co/datasets/thesofakillers/SemCor/resolve/main/data"
SEMCOR_FILES = (
    "brown1-00000-of-00001.csv",
    "brown2-00000-of-00001.csv",
    "brownv-00000-of-00001.csv",
)

#: Treebank tag prefix -> WordNet POS letter.
_TREEBANK_PREFIX_TO_POS = {"N": "n", "V": "v", "J": "a", "R": "r"}

#: Tags that never head a WordNet sense we can score.
_USELESS_POS = {
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "IN",
    "LS",
    "MD",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "TO",
    "UH",
    "WDT",
    "WP",
    "WP$",
    "WRB",
    "SYM",
    "``",
    "''",
    ",",
    ".",
    ":",
    "-LRB-",
    "-RRB-",
    "",
    "None",
}

_POS_FILE = {"n": "data.noun", "v": "data.verb", "a": "data.adj", "r": "data.adv"}


def download_file(url: str, destination: Path) -> Path:
    """Download ``url`` to ``destination`` unless it already exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"  cached: {destination.name}")
        return destination

    print(f"  downloading {destination.name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, destination)  # noqa: S310 - fixed https URLs
    print(f"{destination.stat().st_size // 1024} KB")
    return destination


def fetch_wordnet(cache_dir: Path) -> Path:
    """Download and unpack the WordNet 3.0 database; return the ``dict`` dir."""
    import tarfile

    archive = download_file(WORDNET3_URL, cache_dir / "WNdb-3.0.tar.gz")
    dict_dir = cache_dir / "dict"

    if not dict_dir.exists():
        print("  unpacking WordNet 3.0 ...")
        with tarfile.open(archive) as tar:
            # Defensive: never let an archive write outside the cache directory.
            members = [m for m in tar.getmembers() if not m.name.startswith("/")]
            for member in members:
                target = (cache_dir / member.name).resolve()
                if not str(target).startswith(str(cache_dir.resolve())):
                    raise ValueError(f"Refusing to extract outside cache: {member.name}")
            tar.extractall(cache_dir, members=members)

    return dict_dir


def fetch_semcor(cache_dir: Path) -> list[Path]:
    """Download the SemCor CSVs."""
    return [download_file(f"{SEMCOR_BASE}/{name}", cache_dir / name) for name in SEMCOR_FILES]


def load_sense_to_offset(dict_dir: Path) -> dict[tuple[str, str, int], int]:
    """Build ``(pos, lemma, wn_sense) -> synset offset`` from ``index.sense``.

    ``index.sense`` lines look like::

        enough%5:00:00:sufficient:00 02336109 1 50

    The POS is not stored in that file, so it is recovered from the offset's
    record in the per-POS data files.
    """
    mapping: dict[tuple[str, str, int], int] = {}
    pos_of_offset = _pos_of_offset(dict_dir)

    with open(dict_dir / "index.sense", encoding="latin-1") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2 or not fields[1].strip().isdigit():
                continue

            lemma_part, _, rest = fields[0].partition("%")
            if not rest:
                continue
            bits = rest.split(":")
            if len(bits) < 3 or not bits[1].isdigit():
                continue

            offset = int(fields[1])
            pos = pos_of_offset.get(offset)
            if pos is None:
                continue
            lemma = lemma_part.replace("'", "").replace("_", " ")
            mapping[(pos, lemma, int(bits[1]))] = offset

    return mapping


def _pos_of_offset(dict_dir: Path) -> dict[int, str]:
    """Map synset offset -> POS letter using the per-POS data files."""
    mapping: dict[int, str] = {}
    for pos, filename in _POS_FILE.items():
        with open(dict_dir / filename, encoding="latin-1") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) < 3:
                    continue
                if not (fields[0].strip().isdigit() and fields[1].strip().isdigit()):
                    continue
                mapping[int(fields[0])] = pos
    return mapping


def build_offset_to_sense_id(service: WordNetService) -> dict[tuple[str, int], str]:
    """Build ``(lemma, offset) -> lemma.pos.NN`` using NLTK's own numbering.

    For each lemma, its synsets are listed in canonical order and numbered from 1
    within each POS. Looking the offset up in that list yields the identifier this
    project reports.

    This is delegated to NLTK rather than reimplemented from the WordNet data
    files: numbering senses per lemma is subtle (per lemma *and* POS, adjective
    satellites sharing their parent's numbering, only head words advancing a
    counter), and getting it subtly wrong silently mislabels the corpus.
    """
    mapping: dict[tuple[str, int], str] = {}
    seen: set[str] = set()

    from nltk.corpus import wordnet as wn

    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ")
            if name and name[0].isalpha():
                seen.add(name)

    for lemma in seen:
        try:
            grouped = service.all_synsets_for(lemma)
        except Exception:
            continue
        for pos, group in grouped.items():
            for index, syn in enumerate(group, start=1):
                mapping[(lemma, syn.offset())] = f"{lemma}.{pos}.{index:02d}"

    return mapping


def convert(
    path: Path,
    sense2off: dict[tuple[str, str, int], int],
    offset2id: dict[tuple[str, int], str],
) -> tuple[list[dict[str, object]], int]:
    """Convert one SemCor CSV into evaluation examples."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    tokens = df[df.tag == "wf"].reset_index(drop=True)

    examples: list[dict[str, object]] = []
    unresolved = 0

    for (tagfile, pnum, snum), group in tokens.groupby(["tagfile", "pnum", "snum"], sort=False):
        sentence = " ".join(v for v in group["value"] if v)

        # The engine locates the target by whole-token match, so only lemmas that
        # occur as standalone tokens can ever be predicted. A substring test would
        # let "be" match inside "before" and emit rows that fail at prediction
        # time, which silently shrinks the denominator for one model variant only.
        present = {t.strip(".,!?;:\"'()[]{}").lower() for t in sentence.split()}

        seen: dict[str, int] = {}
        for pos_tag, lemma, lexsn in zip(group["pos"], group["lemma"], group["lexsn"], strict=True):
            if not lexsn or lexsn == "None" or not lemma or lemma == "None":
                continue
            if pos_tag in _USELESS_POS:
                continue

            bits = lexsn.split(":")
            if len(bits) < 2 or not bits[1].isdigit():
                unresolved += 1
                continue

            pos = _TREEBANK_PREFIX_TO_POS.get(pos_tag[0].upper() if pos_tag else "")
            lemma_key = lemma.replace("_", " ")
            if pos is None:
                unresolved += 1
                continue

            offset = sense2off.get((pos, lemma_key, int(bits[1])))
            if offset is None:
                unresolved += 1
                continue

            synset_id = offset2id.get((lemma_key, offset))
            if synset_id is None:
                unresolved += 1
                continue

            # SemCor annotates the lemma while ``value`` is the inflected surface
            # form ("was" for lemma "be"). The gold synset belongs to the lemma,
            # so the lemma is what gets disambiguated, and the occurrence counter
            # tracks lemma occurrences the way the engine counts them.
            if lemma_key.lower() not in present:
                continue

            occurrence = seen.get(lemma_key, 0)
            seen[lemma_key] = occurrence + 1

            examples.append(
                {
                    "sentence": sentence,
                    "target_word": lemma_key,
                    "gold_synset": synset_id,
                    "target_occurrence": occurrence,
                    "pos": synset_id.split(".")[1],
                    "id": f"{tagfile.replace('-', '')}_{pnum}_{snum}_{len(examples)}",
                }
            )

    return examples, unresolved


def drop_unusable(rows: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose gold sense does not exist in WordNet for that lemma.

    A handful of plural lemmas ("eggs", "woods") are annotated in SemCor but do
    not exist as lemmas in WordNet, so no prediction could ever match them.
    """
    from nltk.corpus import wordnet as wn

    def usable(row: pd.Series) -> bool:
        try:
            synset = wn.synset(str(row["gold_synset"]))
        except Exception:
            return False
        lemma = str(row["target_word"]).lower()
        return lemma in {word.name().lower() for word in synset.lemmas()}

    keep = rows.apply(usable, axis=1)
    return rows[keep]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and convert SemCor.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/semcor_brown.csv"),
        help="Destination CSV.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw"),
        help="Where downloads are cached.",
    )
    args = parser.parse_args(argv)

    if not ensure_nltk_resources():
        print("error: NLTK WordNet unavailable; run 'uv run download-resources'", file=sys.stderr)
        return 1

    print("Fetching WordNet 3.0 ...")
    dict_dir = fetch_wordnet(args.cache_dir)
    print("Fetching SemCor ...")
    csvs = fetch_semcor(args.cache_dir)

    print("Building WordNet 3.0 sense mapping ...")
    sense2off = load_sense_to_offset(dict_dir)
    offset2id = build_offset_to_sense_id(WordNetService())
    print(f"  (pos, lemma, wn_sense) -> offset: {len(sense2off)}")
    print(f"  (lemma, offset) -> sense id: {len(offset2id)}")

    rows: list[dict[str, object]] = []
    total_unresolved = 0
    for csv in csvs:
        found, unresolved = convert(csv, sense2off, offset2id)
        total_unresolved += unresolved
        print(f"  {csv.name}: {len(found)} examples ({unresolved} unresolved senses)")
        rows.extend(found)

    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["sentence", "target_word", "target_occurrence", "gold_synset"]
    )
    before = len(frame)
    frame = drop_unusable(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"Wrote {len(frame)} examples to {args.output}")
    print(f"  dropped {before - len(frame)} rows with senses absent from WordNet")
    print(f"  {total_unresolved} unresolved senses skipped during conversion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
