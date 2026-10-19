# ContextSense AI - Architecture

## Module Responsibilities

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                    │
│  - Playground page                                              │
│  - Model comparison page                                        │
│  - Dataset evaluation page                                      │
│  - About & learn page                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Disambiguation Engine                        │
│  contextsense_ai/disambiguation/engine.py                         │
│  - Orchestrates the full WSD pipeline                           │
│  - Selects the appropriate method                                │
│  - Returns structured PredictionResult                           │
└─────────────────────────────────────────────────────────────────┘
         │                                 │
         ▼                                 ▼
┌─────────────────────┐    ┌─────────────────────────────────────┐
│   Baseline Methods  │    │       Context Extractor             │
│  - first_sense      │    │  contextsense_ai/context_extractor.py  │
│  - most_frequent    │    │  - Full-sentence context            │
│                     │    │  - Local window context             │
└─────────────────────┘    │  - Stopword filtering               │
                           │  - Lemmatization options            │
                           └─────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
           ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
           │  WordNet     │    │ Preprocessing│    │  Embeddings  │
           │  Service     │    │              │    │              │
           │              │    │ - Tokenization│   │ Interface    │
           │ - Synset     │    │ - Normalization│  │              │
           │   retrieval  │    │ - Lemmatization│  │ Word2Vec     │
           │ - Relations  │    │ - POS mapping │   │ Backend      │
           └──────────────┘    └──────────────┘   │              │
                                                   │ Sentence     │
                                                   │ Transformer  │
                                                   │ Backend      │
                                                   └──────────────┘
                                                            │
                              ┌─────────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Similarity Calc │
                    │  - Cosine sim    │
                    │  - Safe fallback │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Ranking &       │
                    │  Prediction      │
                    └──────────────────┘
```

## Data Flow

1. **Input**: User provides sentence + target word + occurrence
2. **Preprocessing**: Tokenize, normalize, find target positions
3. **POS filtering**: Detect the target occurrence's part of speech and restrict
   WordNet candidates to it, so a noun sense does not compete against the verb
4. **WordNet lookup**: Get candidate senses for target word, keyed by the target
   word (`bank.n.02`) rather than the synset's first lemma
5. **Context extraction**: Extract context tokens around target
6. **Embedding**: Encode context and sense descriptions to vectors
7. **Similarity**: Compute cosine similarity for each candidate
8. **Ranking**: Sort by similarity score, breaking ties by WordNet order
9. **Output**: Return PredictionResult with ranked candidates

### Two details worth knowing

**Candidate senses are keyed by the target word.** WordNet's `synset.name()` is
built from a synset's *first* lemma, which is not always the word being
disambiguated — the financial sense of *bank* is named
`depository_financial_institution.n.01`. ContextSense reports the conventional
`bank.n.02` instead, so gold labels and predictions are directly comparable.

**POS filtering is a large accuracy lever.** WordNet pools every part of speech
for a lemma, so without filtering, the verb sense *manipulate the mouse* can
outrank the noun sense *a hand-operated device* purely on embedding similarity.
Narrowing candidates by the tagged POS removes that error class entirely.

## Embedding Abstraction

All embedding backends implement `EmbeddingBackend`:

```python
class EmbeddingBackend(ABC):
    @property
    def model_info() -> EmbeddingModelInfo: ...

    def encode_text(text: str) -> EmbeddingResult: ...
    def encode_tokens(tokens: list[str]) -> EmbeddingResult: ...
    def is_available() -> bool: ...
    def load() -> None: ...
    def release() -> None: ...
```

This ensures:
- Context and sense vectors come from the same embedding space
- backends can be swapped without changing the disambiguation logic
- Availability checking is uniform across backends

## WordNet Integration

The `WordNetService` wraps NLTK's WordNet interface:

- `get_candidates(word, pos)` - Retrieve synsets for a word
- `get_synset_from_id(synset_id)` - Resolve a synset ID
- `synset_exists(synset_id)` - Validate a synset ID
- `_build_candidate(syn)` - Build CandidateSense from synset

WordNet is kept independent of the UI and embedding layers.

## Resource Lifecycle

- **NLTK resources**: Downloaded once via `uv run download-resources`
- **Sentence Transformers**: Lazy-loaded on first use, cached in memory
- **Word2Vec**: Must be manually downloaded, loaded on demand. Its cache identity includes the vectors filename, so loading a different file never reuses the previous file's cached sense vectors.
- **Streamlit caching**: Models are held in `st.session_state` rather than behind `st.cache_resource`, so each browser session loads its own copy. Sense vectors are memoised process-wide in `engine._SENSE_VECTOR_CACHE` and shared across sessions.

## UI/Backend Separation

- All NLP logic lives in `src/contextsense_ai/`
- Streamlit UI in `app.py` only handles display and user input
- The `disambiguate()` function is the main entry point from UI to NLP
- Evaluation scripts in `scripts/` use the same backend API
