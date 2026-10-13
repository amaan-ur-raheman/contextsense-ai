"""Context extraction for WSD disambiguation."""

from __future__ import annotations

from contextsense_ai.config import settings
from contextsense_ai.exceptions import DisambiguationError
from contextsense_ai.preprocessing import (
    filter_stopwords,
    find_target_occurrences,
    lemmatize_tokens,
    preprocess_for_target_lookup,
    strip_punctuation_token,
)
from contextsense_ai.schemas import ContextStrategy


class ContextExtractor:
    """Extracts context tokens around a target word occurrence."""

    def __init__(
        self,
        *,
        context_window: int = settings.default_context_window,
        stopword_filtering: bool = False,
        lemmatize: bool = False,
    ) -> None:
        if context_window < settings.min_context_window:
            raise ValueError(f"Context window must be at least {settings.min_context_window}.")
        if context_window > settings.max_context_window:
            raise ValueError(f"Context window must be at most {settings.max_context_window}.")
        self.context_window = context_window
        self.stopword_filtering = stopword_filtering
        self.lemmatize = lemmatize

    def extract(
        self,
        sentence: str,
        target_word: str,
        occurrence_index: int,
        *,
        strategy: ContextStrategy = ContextStrategy.LOCAL_WINDOW,
    ) -> list[str]:
        """Extract context for the target word occurrence.

        Args:
            sentence: The full sentence.
            target_word: The word to disambiguate.
            occurrence_index: Which occurrence to use (0-based).
            strategy: Either FULL_SENTENCE or LOCAL_WINDOW.

        Returns:
            List of context tokens, excluding the selected occurrence.
        """
        preprocessed = preprocess_for_target_lookup(sentence)
        occurrences = find_target_occurrences(preprocessed, target_word)

        if not occurrences:
            raise DisambiguationError(f"Target word '{target_word}' not found in sentence.")

        if occurrence_index >= len(occurrences):
            raise DisambiguationError(
                f"Occurrence index {occurrence_index} out of range. "
                f"Found {len(occurrences)} occurrence(s)."
            )

        target_position = occurrences[occurrence_index]
        tokens = preprocessed.tokens

        if strategy == ContextStrategy.FULL_SENTENCE:
            context_tokens = [t for i, t in enumerate(tokens) if i != target_position]
        else:
            start = max(0, target_position - self.context_window)
            end = min(len(tokens), target_position + self.context_window + 1)
            context_tokens = [
                t for i, t in enumerate(tokens[start:end], start=start) if i != target_position
            ]

        context_tokens = [
            strip_punctuation_token(t) for t in context_tokens if strip_punctuation_token(t)
        ]

        if self.stopword_filtering:
            context_tokens = filter_stopwords(context_tokens)

        if self.lemmatize:
            context_tokens = lemmatize_tokens(context_tokens)

        return context_tokens

    def extract_with_metadata(
        self,
        sentence: str,
        target_word: str,
        occurrence_index: int,
        *,
        strategy: ContextStrategy = ContextStrategy.LOCAL_WINDOW,
    ) -> dict[str, object]:
        """Extract context and return metadata for display."""
        preprocessed = preprocess_for_target_lookup(sentence)
        occurrences = find_target_occurrences(preprocessed, target_word)

        if not occurrences:
            raise DisambiguationError(f"Target word '{target_word}' not found in sentence.")

        if occurrence_index >= len(occurrences):
            raise DisambiguationError(
                f"Occurrence index {occurrence_index} out of range. "
                f"Found {len(occurrences)} occurrence(s)."
            )

        target_position = occurrences[occurrence_index]
        all_tokens = preprocessed.tokens
        context_tokens = self.extract(sentence, target_word, occurrence_index, strategy=strategy)

        return {
            "context_tokens": context_tokens,
            "all_tokens": all_tokens,
            "target_position": target_position,
            "occurrences": occurrences,
            "selected_occurrence": occurrence_index,
            "highlighted_sentence": preprocessed.highlight_token(target_position),
            "strategy": strategy.value,
            "context_window": self.context_window,
        }
