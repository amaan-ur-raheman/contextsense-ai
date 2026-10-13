"""WordNet service for retrieving candidate senses."""

from __future__ import annotations

from typing import cast

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from contextsense_ai.exceptions import WordNetError
from contextsense_ai.schemas import CandidateSense

_SENSES_PER_RELATION = 10


def synsets_for_word(word: str) -> list[Synset]:
    """Helper to get all synsets for a word."""
    return cast(list[Synset], wn.synsets(word))


class WordNetService:
    """Service for retrieving WordNet candidate senses for a target word.

    Instances are cached; the underlying NLTK corpus reader is a process-wide
    resource, so there is no benefit to constructing one per request.
    """

    _instance: WordNetService | None = None

    def __new__(cls) -> WordNetService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._sense_index_cache: dict[tuple[str, str], dict[str, str]] = {}
        self._initialized = True
        self._ensure_resources()

    def _ensure_resources(self) -> None:
        """Ensure required NLTK resources are available."""
        from contextsense_ai.utils.resource_manager import ensure_nltk_resources

        if not ensure_nltk_resources():
            raise WordNetError(
                "NLTK WordNet resource not found. Run: "
                "uv run python -m contextsense_ai.scripts.download_resources"
            )

        try:
            wn.synsets("test")
        except Exception as exc:
            raise WordNetError(
                "NLTK WordNet resource not found. Run: "
                "uv run python -m contextsense_ai.scripts.download_resources"
            ) from exc

    # ------------------------------------------------------------------ ids

    def sense_id_for(self, syn: Synset, word: str) -> str:
        """Return the sense ID of ``syn`` relative to the target word.

        WordNet's own ``synset.name()`` is built from a synset's *first* lemma,
        which is not necessarily the word being disambiguated. That makes
        ``depository_financial_institution.n.01`` the reported ID for the
        financial sense of "bank" (conventionally ``bank.n.02``), so gold
        labels could never match a prediction.

        This reconstructs the conventional ``word.pos.NN`` numbering by
        indexing the synsets that contain ``word``, restricted to the synset's
        own POS. For senses where the target already is the first lemma this
        reproduces WordNet's ``name()`` exactly.
        """
        word = word.strip().lower()
        cache_key = (word, syn.pos())
        index = self._sense_index_cache.get(cache_key)
        if index is None:
            index = {
                s.offset(): f"{word}.{s.pos()}.{i:02d}"
                for i, s in enumerate(
                    (s for s in wn.synsets(word) if s.pos() == syn.pos()),
                    start=1,
                )
            }
            self._sense_index_cache[cache_key] = index
        return index.get(syn.offset()) or str(syn.name())

    def get_synset_from_id(self, synset_id: str) -> Synset | None:
        """Get a Synset object from its ID string.

        ID format: lemma.pos.sense_number (e.g. ``bank.n.01``).

        NLTK's ``wn.synset()`` resolves unknown sense numbers by falling back to
        the lemma, so ``spring.n.99`` silently yields some unrelated synset. The
        resolved synset is therefore checked to confirm it actually carries the
        requested lemma.
        """
        if not synset_id:
            return None
        try:
            syn = wn.synset(synset_id)
        except Exception:
            return None

        lemma = synset_id.split(".", 1)[0].lower()
        if not any(lem.name().lower() == lemma for lem in syn.lemmas()):
            return None
        return syn

    def synset_exists(self, synset_id: str) -> bool:
        """Check whether a synset ID refers to a valid WordNet synset."""
        return self.get_synset_from_id(synset_id) is not None

    def get_definition(self, synset_id: str) -> str | None:
        """Retrieve the definition for a given synset ID, or None if invalid."""
        syn = self.get_synset_from_id(synset_id)
        return None if syn is None else syn.definition()

    # ------------------------------------------------------------ candidates

    def get_candidates(
        self,
        word: str,
        *,
        pos: str | None = None,
        synset_offset: str | None = None,
    ) -> list[CandidateSense]:
        """Retrieve candidate senses for a word, optionally filtered by POS.

        Args:
            word: The target word to look up.
            pos: Optional part of speech filter ('n', 'v', 'a', 'r').
            synset_offset: Optional synset offset to filter to a single sense.

        Returns:
            Candidates in WordNet's order, each carrying a target-relative
            ``synset_id`` and a SemCor frequency rank.
        """
        word = word.strip().lower()
        if not word:
            raise WordNetError("Target word must not be empty.")

        pos_filter = self._normalize_pos(pos)

        try:
            synsets = wn.synsets(word)
        except Exception as exc:
            raise WordNetError(f"Failed to retrieve synsets for '{word}'.") from exc

        candidates: list[CandidateSense] = []
        seen_offsets: set[str] = set()

        for syn in synsets:
            if pos_filter is not None and syn.pos() != pos_filter:
                continue

            offset = syn.offset()
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)

            candidates.append(self._build_candidate(syn, word))

            if synset_offset and offset == synset_offset:
                break

        self._assign_frequency_ranks(candidates, word)
        return candidates

    def _normalize_pos(self, pos: str | None) -> str | None:
        """Normalize POS tag to WordNet format."""
        if pos is None:
            return None
        mapping = {
            "noun": "n",
            "verb": "v",
            "adjective": "a",
            "adj": "a",
            "adverb": "r",
            "adv": "r",
        }
        return mapping.get(pos.lower(), pos)

    def _build_candidate(self, syn: Synset, word: str) -> CandidateSense:
        """Build a CandidateSense from a WordNet synset."""
        lemmas = list({lem.name().replace("_", " ") for lem in syn.lemmas()})
        examples = [ex for ex in syn.examples()]
        hypernyms = self._get_related_names(syn, "hypernyms")
        hyponyms = self._get_related_names(syn, "hyponyms")

        warnings: list[str] = []
        if not examples:
            warnings.append("No example sentences available for this sense.")

        return CandidateSense(
            synset=syn,
            pos=syn.pos(),
            lemmas=lemmas,
            definition=syn.definition(),
            examples=examples,
            hypernyms=hypernyms,
            hyponyms=hyponyms,
            synset_id=self.sense_id_for(syn, word),
            warnings=warnings,
        )

    def _get_related_names(self, syn: Synset, relation: str) -> list[str]:
        """Get names of related synsets for a given relation."""
        try:
            related = getattr(syn, relation)()
        except Exception:
            return []
        names: list[str] = []
        for s in related[:_SENSES_PER_RELATION]:
            lemma_names = [lem.name().replace("_", " ") for lem in s.lemmas()]
            if lemma_names:
                names.append(lemma_names[0])
        return names

    # ------------------------------------------------------------ frequency

    def _semcor_count(self, syn: Synset, word: str) -> int:
        """SemCor corpus frequency of ``word`` within ``syn``.

        Requires the WordNet 3.0 sense-disambiguated corpus. Returns 0 when the
        corpus is unavailable or the word does not occur in this sense.
        """
        try:
            return sum(lem.count() for lem in syn.lemmas() if lem.name().lower() == word.lower())
        except Exception:
            return 0

    def _assign_frequency_ranks(self, candidates: list[CandidateSense], word: str) -> None:
        """Assign 1-based frequency ranks to candidates, most frequent first.

        Uses the SemCor count attached to each WordNet lemma. Candidates with a
        count of 0 are unranked (``None``) because the corpus is small enough
        that absence is far more likely to mean "not attested" than "rare".

        Ties are broken by WordNet's own ordering, which is the conventional
        frequency heuristic.
        """
        scored: list[tuple[int, int, CandidateSense]] = []
        for index, candidate in enumerate(candidates):
            count = self._semcor_count(candidate.synset, word)
            candidate.frequency_count = count
            scored.append((count, index, candidate))

        scored.sort(key=lambda item: (-item[0], item[1]))

        rank = 0
        previous_count: int | None = None
        for count, _, candidate in scored:
            if count <= 0:
                candidate.frequency_rank = None
                continue
            if count != previous_count:
                rank += 1
                previous_count = count
            candidate.frequency_rank = rank
