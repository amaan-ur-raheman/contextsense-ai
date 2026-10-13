# ContextSense AI - Viva Questions and Answers

This document contains viva/questions and answers for defending the ContextSense AI project. All answers describe the actual implementation.

## Fundamentals

### 1. What is Word Sense Disambiguation (WSD)?

Word Sense Disambiguation is the task of determining which meaning (sense) of an ambiguous word is being used in a particular context. For example, "bank" can mean a financial institution or land beside a river. WSD systems must choose the correct sense based on context.

### 2. What is lexical ambiguity?

Lexical ambiguity occurs when a word has multiple possible meanings. It is the fundamental problem WSD addresses. For example, "bat" can mean a flying mammal or a sports implement. The context determines which meaning is intended.

### 3. What is homonymy versus polysemy?

Homonymy: Words that share spelling/pronunciation but have unrelated meanings (e.g., "bank" as financial institution vs. river bank). Polysemy: A single word with multiple related meanings (e.g., "head" of a person vs. "head" of a company). WordNet treats both as separate senses.

### 4. What is the difference between coarse-grained and fine-grained WSD?

Coarse-grained WSD groups similar senses together (e.g., all "bank" financial senses as one). Fine-grained WSD distinguishes between every WordNet synset. ContextSense AI performs fine-grained WSD using WordNet synset IDs.

## WordNet

### 5. What is WordNet?

WordNet is a large lexical database of English developed at Princeton University. It groups words into sets of synonyms called synsets and records various semantic relations between them. It is widely used in NLP research.

### 6. What is a synset?

A synset (synonym set) is a set of synonyms that represent a single distinct concept. For example, the synset `bank.n.01` contains {bank, banking institution, financial institution}. Each synset has a unique ID, definition, and examples.

### 7. What are the main semantic relations in WordNet?

- **Hypernyms**: More general concepts (is-a relationship). Example: "animal" is a hypernym of "dog".
- **Hyponyms**: More specific concepts. Example: "dog" is a hyponym of "animal".
- **Meronyms**: Part-of relationships. Example: "wheel" is a meronym of "car".
- **Holonyms**: Has-part relationships. Example: "car" is a holonym of "wheel".
- **Antonyms**: Opposite meanings (for adjectives and adverbs).

### 8. What is a synset ID and what is its format?

A synset ID has the format `lemma.pos.offset`:
- `lemma`: A representative word for the sense
- `pos`: Part of speech code (n=noun, v=verb, a=adjective, r=adverb)
- `offset`: A unique numeric identifier

Example: `bank.n.01` refers to the first noun sense of "bank".

### 9. How does ContextSense AI use WordNet?

The `WordNetService` class retrieves candidate synsets for a target word, extracts their definitions, examples, lemmas, and semantic relations (hypernyms, hyponyms), and builds `CandidateSense` objects. These candidates form the set of possible meanings that the system ranks.

### 10. What POS tags does the system support?

The system supports nouns (n), verbs (v), adjectives (a), and adverbs (r). Users can optionally filter candidates by part of speech in the UI.

## Embeddings

### 11. What is an embedding?

An embedding is a dense numerical vector that represents a word or text in a continuous vector space. Similar meanings produce similar vectors (close together in the space). Embeddings capture semantic relationships that one-hot encoding cannot.

### 12. What is one-hot encoding and why is it limited?

One-hot encoding represents each word as a sparse vector with a 1 in the position corresponding to that word and 0s elsewhere. Limitations: (1) extremely high-dimensional (vocabulary size), (2) all words are orthogonal (no notion of similarity), (3) cannot capture semantic relationships, (4) doesn't scale to large vocabularies.

### 13. What is Word2Vec?

Word2Vec is a method for learning word embeddings introduced by Mikolov et al. (2013). It learns dense vector representations by predicting words from their context. Word2Vec produces static embeddings: the same word always gets the same vector regardless of context.

### 14. What are CBOW and Skip-gram?

CBOW (Continuous Bag of Words): Predicts a target word from its surrounding context words. Faster to train, works well for frequent words.

Skip-gram: Predicts surrounding context words given a target word. Slower but better for rare words.

ContextSense AI's Word2Vec backend uses pretrained Google News vectors, not training its own model.

### 15. What is a static embedding versus a contextual embedding?

Static embeddings (Word2Vec, GloVe): Each word has a single fixed vector regardless of context. The word "bank" has the same vector in "river bank" and "bank account".

Contextual embeddings (BERT, Sentence Transformers): The vector for a word depends on its context. The word "bank" gets different vectors in different sentences. This is more powerful for WSD.

### 16. What is Sentence Transformers?

Sentence Transformers (Reimers & Gurevych, 2019) is a framework for producing dense embeddings of sentences using transformer models. It extends models like BERT to produce meaningful sentence-level representations. ContextSense AI uses the `all-MiniLM-L6-v2` model by default (384 dimensions, lightweight).

### 17. What is mean pooling?

Mean pooling is a method to combine multiple vectors into a single vector by taking their average:

```
pooled_vector = (v1 + v2 + ... + vn) / n
```

In ContextSense AI, mean pooling is used in the Word2Vec backend to combine word vectors into a single context vector or sense description vector.

### 18. What are the limitations of mean pooling?

- Loses word order information
- Treats all words equally (no attention to important words)
- Cannot model interactions between words
- Out-of-vocabulary words are silently dropped
- The resulting vector may not adequately represent the full meaning

### 19. What is an out-of-vocabulary (OOV) word?

An OOV word is a word that is not in the embedding model's vocabulary. The model has no vector for it. In ContextSense AI's Word2Vec backend, OOV words are skipped during mean pooling. This can result in poorer representations if many context words are OOV.

### 20. Which embedding backend does ContextSense AI use by default?

The system uses Sentence Transformers (`all-MiniLM-L6-v2`) by default. This model auto-downloads on first use and provides contextual embeddings without requiring manual download of a large model. Word2Vec is available as an optional backend but requires manual download of the ~3.6GB Google News model.

## Similarity and Scoring

### 21. What is cosine similarity?

Cosine similarity measures the cosine of the angle between two vectors:

```
cos(A, B) = (A · B) / (||A|| × ||B||)
```

It ranges from -1 (opposite direction) to 1 (identical direction), with 0 meaning orthogonal (no similarity). It is independent of vector magnitude, focusing only on direction.

### 22. Why use cosine similarity for WSD?

Cosine similarity is appropriate because:
- It measures directional alignment, not magnitude
- It works in high-dimensional spaces
- It is bounded and interpretable
- It is the standard similarity measure for embedding-based WSD
- It can compare context vectors with sense description vectors

### 23. Is a similarity score a probability?

No. A cosine similarity score is NOT a probability. It indicates how closely two vectors align in direction, not the likelihood that a prediction is correct. A score of 0.8 does not mean "80% chance this is right." The system explicitly states this in its UI and documentation.

### 24. What is a context vector?

A context vector is the embedding representation of the words surrounding the target word. It is created by encoding the context tokens (using Sentence Transformers or Word2Vec mean pooling) into a single vector. The context vector represents the "meaning" of the surrounding text.

### 25. What is a sense vector?

A sense vector is the embedding representation of a candidate sense's description. It is created by encoding the WordNet definition, examples, lemmas, and relations into a vector. The sense vector represents what that particular meaning "means."

### 26. How does the system select the predicted sense?

After computing similarity scores for all candidates, the system sorts them in descending order and selects the top-ranked candidate (rank 1). The synset ID and definition of this candidate become the prediction.

## Baselines

### 27. What is a baseline in machine learning?

A baseline is a simple method that serves as a reference point for evaluating more sophisticated approaches. A good baseline should be: (1) easy to implement, (2) computationally cheap, (3) representative of a naive or minimal approach, (4) useful for showing that a complex method adds value.

### 28. What baselines does ContextSense AI implement?

1. **First-sense baseline**: Always picks the first WordNet sense for the target word. Does not use context.
2. **Most-frequent-sense baseline**: Picks the sense with the highest estimated frequency using WordNet lemma counts. Falls back to first sense when frequency data is unavailable.

### 29. Why is the most-frequent-sense baseline useful?

The most-frequent-sense (MFS) baseline is often surprisingly strong in WSD. Many words are used in their most common sense most of the time. MFS provides a realistic lower bound: any WSD system should outperform simply guessing the most common sense. If a system cannot beat MFS, it may not be adding value.

### 30. Does the first WordNet sense always equal the most frequent sense?

No. WordNet's first sense is not guaranteed to be the most frequent sense. WordNet's ordering is based on various factors. The system explicitly warns users about this in its UI and documentation.

## Evaluation

### 31. How is the model evaluated?

The system evaluates on labelled examples where each example has a sentence, target word, and gold synset ID. It runs disambiguation on each example and compares the predicted synset to the gold label. Metrics are computed using scikit-learn.

### 32. What metrics are reported?

- **Accuracy**: Fraction of correct predictions
- **Precision (macro)**: Average precision across all senses
- **Recall (macro)**: Average recall across all senses
- **F1 (macro)**: Harmonic mean of macro precision and recall
- **F1 (weighted)**: F1 weighted by sense frequency
- **Per-class metrics**: Performance for each individual sense

### 33. What is the difference between precision and recall?

Precision: Of all the times we predicted sense X, how many were correct? High precision means few false positives.

Recall: Of all the times sense X was the correct answer, how many did we predict? High recall means few false negatives.

F1 combines both into a single balanced measure.

### 34. What is a gold label?

A gold label is the correct, annotated answer for an evaluation example. In WSD, the gold label is the WordNet synset ID that human annotators assigned to the target word in that context. The system compares its predictions to gold labels to compute metrics.

### 35. What datasets does the system use for evaluation?

The system includes:
1. **Sample dataset**: 8 built-in demonstration examples (NOT a benchmark)
2. **CSV upload**: Users can upload their own labelled datasets
3. **SemCor support**: Code is provided to load SemCor when manually downloaded

The sample dataset should never be used to report benchmark performance.

### 36. How does the system handle skipped examples?

Skipped examples (invalid data, model failures, empty context) are reported separately. They are NOT included in the denominator when computing accuracy. The system reports: total examples, valid examples, evaluated examples, skipped examples, and invalid examples.

### 37. Why report per-class metrics?

Per-class metrics reveal how the model performs on individual senses. A model might have good overall accuracy by always predicting common senses, but perform poorly on rare senses. Per-class metrics expose this. They show precision, recall, F1, and support for each sense.

## System Design

### 38. What is the system architecture?

The system has clear separation between:
- **UI layer** (Streamlit app): Handles user interaction
- **Disambiguation engine**: Orchestrates the WSD pipeline
- **Embedding backends**: Provide vector representations (abstracted via interface)
- **WordNet service**: Provides candidate senses
- **Context extractor**: Extracts context tokens
- **Evaluation framework**: Computes metrics and runs experiments

### 39. How does the embedding interface work?

All embedding backends implement the `EmbeddingBackend` abstract base class with methods: `encode_text()`, `encode_tokens()`, `is_available()`, `load()`, `release()`, and `model_info`. This allows swapping backends without changing the disambiguation logic. Context and sense vectors always come from the same backend.

### 40. How does the system handle model loading?

Models are loaded lazily on first use, not at startup. The Sentence Transformer model is cached in memory after loading. Streamlit's `st.cache_resource` pattern is used to avoid reloading on every UI interaction. The Word2Vec model requires manual download and is loaded on demand.

## Limitations

### 41. What are the main limitations of this system?

1. **Context window**: Local window may miss relevant long-range context
2. **Sense descriptions**: WordNet definitions may not distinguish closely related senses
3. **Mean pooling**: Loses word order and treats all words equally
4. **Static embeddings**: Word2Vec cannot adapt to context
5. **No syntactic parsing**: Does not use dependency relations
6. **Small demo dataset**: Sample examples are not a benchmark
7. **No training**: The system uses pretrained models, not task-specific training

### 42. Why might Sentence Transformers fail for WSD?

- The model was not specifically trained for WSD
- Sense descriptions may be too short or ambiguous
- Related senses may have very similar descriptions
- The model may not capture fine-grained sense distinctions
- Network issues during model download
- Memory constraints on smaller machines

### 43. What would improve this system?

1. Dependency parsing for better context selection
2. Fine-tuning a transformer on WSD data
3. Training dedicated sense embeddings
4. Using a larger, properly annotated benchmark (SemCor)
5. Ensemble methods combining multiple approaches
6. Incorporating POS tagging and syntactic features
7. Domain-specific adaptations

### 44. Can this system be used in production?

Not without further development. This is an educational system demonstrating WSD concepts. Production use would require: larger annotated training data, task-specific model training, more robust error handling, comprehensive evaluation on standard benchmarks, and careful accuracy validation for the target domain.

## Technical Implementation

### 45. What Python packages does the system use?

- **NLTK**: WordNet access, tokenization
- **Streamlit**: Web interface
- **Sentence Transformers**: Contextual embeddings
- **Gensim**: Word2Vec model loading
- **NumPy**: Vector operations
- **scikit-learn**: Evaluation metrics
- **Pandas**: Data handling
- **Plotly**: Optional visualization
- **pytest**: Testing
- **Ruff**: Linting

### 46. How does the system validate target word occurrences?

The preprocessing module tokenizes the sentence, finds all occurrences of the target word (case-insensitive, handling punctuation), and allows the user to select which occurrence to disambiguate. If the word appears multiple times, the user must select the intended one.

### 47. How are similarity scores validated?

The cosine similarity function validates: (1) vectors are non-empty, (2) dimensions match, (3) values are finite (no NaN/Inf), (4) neither vector has zero norm. If validation fails, a safe fallback returns (False, None, error_message) instead of raising an exception, allowing graceful handling.

### 48. What happens when no candidates can be scored?

If all candidate senses fail to produce valid similarity scores (due to encoding errors, empty vectors, etc.), the system returns a failure result with an explanatory error message. It does NOT silently pick a random sense or default to a fabricated answer.

### 49. How does the system ensure reproducibility?

- Fixed random seed (42) in configuration
- Exact model names are recorded in evaluation results
- Per-example predictions are exported
- Configuration (context window, strategy, preprocessing) is logged
- The same input always produces the same output for a given model and backend state

### 50. What is the role of the explanation module?

The explanation module generates step-by-step explanations of predictions for educational purposes. It shows: how many candidates were found, what context was used, how vectors were created, how similarity was calculated, and why the top candidate was selected. This is useful for teaching and viva demonstrations.
