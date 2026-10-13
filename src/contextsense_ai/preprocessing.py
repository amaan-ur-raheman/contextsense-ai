"""Text preprocessing utilities for tokenization, normalization, and lemmatization."""

from __future__ import annotations

import re
from dataclasses import dataclass

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

from contextsense_ai.exceptions import PreprocessingError

_TOKENIZER = TreebankWordTokenizer()
_LEMMATIZER = WordNetLemmatizer()

# ---------- basic tokenization/normalization -------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Tokenize text using TreebankWordTokenizer."""
    if not text:
        raise PreprocessingError("Cannot tokenize empty text.")
    return list(_TOKENIZER.tokenize(text))


def normalize_case(tokens: list[str]) -> list[str]:
    """Lowercase every token."""
    return [token.lower() for token in tokens]


def strip_punctuation_token(token: str) -> str:
    """Remove leading/trailing punctuation from a single token."""
    return token.strip(".,!?;:\"'()[]{}")


def normalize_text(text: str) -> str:
    """Normalize whitespace and Unicode in text."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- stopword filtering ----------------------------------------------------------------


DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "as",
        "until",
        "while",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "any",
        "s",
        "t",
        "don",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "couldn",
        "didn",
        "doesn",
        "hadn",
        "hasn",
        "haven",
        "isn",
        "ma",
        "mightn",
        "mustn",
        "needn",
        "shan",
        "shouldn",
        "wasn",
        "weren",
        "won",
        "wouldn",
    }
)


def filter_stopwords(
    tokens: list[str],
    *,
    stopwords: frozenset[str] | None = None,
) -> list[str]:
    """Filter out stopwords, preserving negation words by default."""
    stop_set = stopwords or DEFAULT_STOPWORDS
    return [t for t in tokens if t.lower() not in stop_set]


# ---------- lemmatization ---------------------------------------------------------------------


def get_wordnet_pos(treebank_tag: str) -> str:
    """Map Treebank POS tag to WordNet POS."""
    if treebank_tag.startswith("J"):
        return str(wordnet.ADJ)
    if treebank_tag.startswith("V"):
        return str(wordnet.VERB)
    if treebank_tag.startswith("N"):
        return str(wordnet.NOUN)
    if treebank_tag.startswith("R"):
        return str(wordnet.ADV)
    return str(wordnet.NOUN)


def lemmatize_token(token: str, pos: str = "n") -> str:
    """Lemmatize a single token using WordNet lemmatizer."""
    return str(_LEMMATIZER.lemmatize(token.lower(), pos=pos))


def lemmatize_tokens(
    tokens: list[str],
    *,
    pos_tagged: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Lemmatize a list of tokens, optionally using POS tags."""
    if pos_tagged is None:
        return [_LEMMATIZER.lemmatize(t.lower(), pos="n") for t in tokens]
    result: list[str] = []
    for token, tag in pos_tagged:
        wn_pos = get_wordnet_pos(tag)
        result.append(_LEMMATIZER.lemmatize(token.lower(), pos=wn_pos))
    return result


# ---------- part-of-speech tagging ---------------------------------------------------------------

_TAGGER_READY = False

_TREEBANK_TO_WORDNET = {
    "N": "n",
    "V": "v",
    "J": "a",
    "R": "r",
}


def _ensure_tagger() -> bool:
    """Return True if a POS tagger is available, downloading it once if needed."""
    global _TAGGER_READY
    if _TAGGER_READY:
        return True

    from contextsense_ai.utils.resource_manager import ensure_nltk_resources

    if not ensure_nltk_resources():
        return False

    try:
        from nltk import pos_tag

        pos_tag(["test"])
        _TAGGER_READY = True
    except Exception:
        _TAGGER_READY = False
    return _TAGGER_READY


def get_wordnet_pos_from_tag(treebank_tag: str) -> str | None:
    """Map a Treebank POS tag to a WordNet POS, or None if not mappable.

    ``None`` means "no opinion" (for example an adverb used as a modifier),
    which callers should treat as "do not filter by POS".
    """
    if not treebank_tag or not treebank_tag[0].isalpha():
        return None
    return _TREEBANK_TO_WORDNET.get(treebank_tag[0].upper())


def tag_target_pos(sentence: str, target_word: str, occurrence_index: int) -> str | None:
    """Return the WordNet POS of a specific occurrence of ``target_word``.

    Returns None when the word is absent, the occurrence is out of range, or no
    tagger is installed — callers then fall back to considering all parts of
    speech rather than filtering incorrectly.
    """
    preprocessed = preprocess_for_target_lookup(sentence)
    occurrences = find_target_occurrences(preprocessed, target_word)
    if not occurrences or occurrence_index >= len(occurrences):
        return None

    if not _ensure_tagger():
        return None

    try:
        from nltk import pos_tag, word_tokenize

        tagged = pos_tag(word_tokenize(preprocessed.original))
    except Exception:
        return None

    # word_tokenize and TreebankWordTokenizer agree closely enough on the
    # leading tokens; match by token text from the occurrence onwards so that
    # any disagreement in earlier punctuation does not shift the index.
    target = preprocessed.tokens[occurrences[occurrence_index]]
    for index, (token, tag) in enumerate(tagged):
        if token == target and index >= occurrences[occurrence_index]:
            return get_wordnet_pos_from_tag(tag)
    return None


# ---------- token position mapping ------------------------------------------------------------


@dataclass
class TokenSpan:
    """Location of a token in the original text."""

    start: int
    end: int
    token: str
    normalized: str
    index: int


@dataclass
class PreprocessedSentence:
    """Preprocessed sentence with token spans for highlighting."""

    original: str
    tokens: list[str]
    normalized_tokens: list[str]
    spans: list[TokenSpan]
    token_to_original_index: list[int]

    def highlight_token(self, occurrence_index: int) -> str:
        """Return the original sentence with the target occurrence highlighted."""
        if occurrence_index >= len(self.spans):
            raise PreprocessingError(
                f"Occurrence index {occurrence_index} out of range "
                f"(found {len(self.spans)} tokens)."
            )
        span = self.spans[occurrence_index]
        before = self.original[: span.start]
        target = self.original[span.start : span.end]
        after = self.original[span.end :]
        return f"{before}**{target}**{after}"


def preprocess_for_target_lookup(
    sentence: str,
    *,
    target_word: str | None = None,
) -> PreprocessedSentence:
    """Preprocess a sentence for target-word lookup, preserving spans."""
    if not sentence or not sentence.strip():
        raise PreprocessingError("Sentence must not be empty.")

    original = normalize_text(sentence)
    tokens = tokenize(original)
    spans: list[TokenSpan] = []
    pos = 0
    for idx, tok in enumerate(tokens):
        start = original.find(tok, pos)
        if start == -1:
            start = pos
        end = start + len(tok)
        spans.append(
            TokenSpan(
                start=start,
                end=end,
                token=tok,
                normalized=tok.lower(),
                index=idx,
            )
        )
        pos = end

    normalized_tokens = [s.normalized for s in spans]
    token_to_original_index = list(range(len(tokens)))

    return PreprocessedSentence(
        original=original,
        tokens=tokens,
        normalized_tokens=normalized_tokens,
        spans=spans,
        token_to_original_index=token_to_original_index,
    )


def find_target_occurrences(
    preprocessed: PreprocessedSentence,
    target_word: str,
) -> list[int]:
    """Find indices of all occurrences of target_word in preprocessed tokens."""
    target_lower = target_word.lower().strip()
    if not target_lower:
        raise PreprocessingError("Target word must not be empty.")
    occurrences: list[int] = []
    for idx, norm in enumerate(preprocessed.normalized_tokens):
        stripped = strip_punctuation_token(norm)
        if stripped == target_lower:
            occurrences.append(idx)
    return occurrences


def find_target_occurrences_raw(
    sentence: str,
    target_word: str,
) -> list[int]:
    """Find occurrences of target_word in a raw sentence (simple whitespace tokenization)."""
    tokens = sentence.split()
    target_lower = target_word.lower().strip()
    occurrences: list[int] = []
    for idx, tok in enumerate(tokens):
        stripped = tok.strip(".,!?;:\"'()[]{}")
        if stripped.lower() == target_lower:
            occurrences.append(idx)
    return occurrences
