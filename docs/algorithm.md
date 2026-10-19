# ContextSense AI - Algorithm

## Overview

The disambiguation algorithm combines lexical knowledge from WordNet with
embedding-based semantic similarity to predict the intended sense of an
ambiguous word in context.

## Inputs

- **sentence**: The full text containing the target word
- **target_word**: The word to disambiguate
- **occurrence_index**: Which occurrence to use (0-based)
- **model_variant**: Which WSD method to use
- **context_strategy**: How to extract context (full_sentence or local_window)
- **context_window**: Size of local context window (if applicable)

## Algorithm Steps

### Step 1: Preprocessing

```
1. Normalize whitespace in the sentence
2. Tokenize using TreebankWordTokenizer
3. Build token spans mapping back to original text
4. Find all occurrences of target_word (case-insensitive)
5. Validate that target_word exists in the sentence
6. Select the specified occurrence
```

### Step 2: Candidate Generation (WordNet)

```
For each synset of the target word:
    1. Get synset from WordNet
    2. Extract:
       - Synset ID (lemma.pos.sense_number, relative to the target word)
       - Part of speech
       - Lemmas (synonym set)
       - Definition (gloss)
       - Example sentences
       - Hypernyms (is-a relations)
       - Hyponyms (has-type relations)
    3. Build a CandidateSense object
```

### Step 3: Context Extraction

**Full sentence strategy:**
```
1. Use all tokens except the target occurrence
2. Filter punctuation
3. Optionally filter stopwords
4. Optionally lemmatize
```

**Local window strategy:**
```
1. Identify target token position
2. Take window tokens on each side:
   start = max(0, position - window_size)
   end = min(len(tokens), position + window_size + 1)
3. Exclude the target token itself
4. Filter punctuation
5. Optionally filter stopwords
6. Optionally lemmatize
```

### Step 4: Context Representation

**Sentence Transformers backend:**
```
1. Join context tokens into a text string
2. Encode with SentenceTransformer model
3. Result: dense vector (e.g., 384D for all-MiniLM-L6-v2)
```

**Word2Vec backend:**
```
1. Tokenize context text
2. Look up each token in Word2Vec vocabulary
3. Collect available vectors (skip OOV tokens)
4. Compute mean pooling:
   context_vector = mean(available_vectors)
5. Result: 300D vector
```

### Step 5: Sense Representation

For each candidate sense:

**Sentence Transformers backend:**
```
1. Build sense description:
   description = definition + examples + lemmas + relations
2. Encode description text with SentenceTransformer
3. Result: dense vector
```

**Word2Vec backend:**
```
1. Tokenize sense description
2. Look up each token in Word2Vec vocabulary
3. Compute mean pooling over available vectors
4. Result: 300D vector
```

### Step 6: Similarity Calculation

For each candidate sense:

```
1. Compute cosine similarity between context vector and sense vector:
   similarity = dot(context_vec, sense_vec) / (norm(context_vec) × norm(sense_vec))

2. Validate:
   - Both vectors non-empty
   - Dimensions match
   - Values are finite
   - Neither vector has zero norm

3. Store similarity score with the candidate
```

### Step 7: Ranking and Prediction

```
1. Sort candidates by similarity score (descending)
2. Assign ranks (1 = highest similarity)
3. Select the top-ranked candidate as the prediction
4. Return structured PredictionResult
```

## Baseline Methods

### First-Sense Baseline
```
1. Get WordNet candidates for target word
2. Return the first candidate as the prediction
3. No context or embeddings used
```

### Most-Frequent-Sense Baseline
```
1. Get WordNet candidates with frequency information
2. Sort by estimated frequency (lemma counts)
3. Return the most frequent sense
4. Fall back to first sense if frequency unavailable
```

## Cosine Similarity Formula

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

where:
- A · B = sum(a_i × b_i) for all dimensions i
- ||A|| = sqrt(sum(a_i²)) (Euclidean norm)
- ||B|| = sqrt(sum(b_i²))
```

**Properties:**
- Range: [-1, 1]
- 1: identical direction (maximum similarity)
- 0: orthogonal (no similarity)
- -1: opposite direction
- Symmetric: cos(A, B) = cos(B, A)

## Pseudocode

```
function disambiguate(sentence, target_word, occurrence_idx, model_variant, embedding_backend, context_config):
    # Step 1: Get candidates
    candidates = wordnet_service.get_candidates(target_word)

    if not candidates:
        return failure("No senses found for target word")

    # Step 2: Baseline methods
    if model_variant == FIRST_SENSE:
        return first_sense_baseline(candidates)

    if model_variant == MOST_FREQUENT_SENSE:
        return most_frequent_sense_baseline(candidates)

    # Step 3: Extract context
    context_tokens = context_extractor.extract(
        sentence, target_word, occurrence_idx,
        strategy=context_config.strategy,
        window=context_config.window
    )

    if not context_tokens:
        return failure("Empty context")

    # Step 4: Encode context
    try:
        context_vector = embedding_backend.encode_text(
            join(context_tokens)
        )
    except EmbeddingError as e:
        return failure(f"Context encoding failed: {e}")

    # Step 5: Encode senses and compute similarity
    ranked = []
    for candidate in candidates:
        try:
            sense_description = build_description(candidate)
            sense_vector = embedding_backend.encode_text(sense_description)
            score = cosine_similarity(context_vector, sense_vector)
            ranked.append((candidate, score))
        except EmbeddingError:
            continue  # Skip this candidate

    if not ranked:
        return failure("No candidates could be scored")

    # Step 6: Rank and predict
    ranked.sort(key=lambda x: x[1], reverse=True)
    prediction = ranked[0]

    return PredictionResult(
        sentence=sentence,
        target_word=target_word,
        context=context_tokens,
        candidates=candidates,
        ranked_candidates=ranked,
        predicted_synset=prediction[0].synset_id,
        predicted_definition=prediction[0].definition,
        top_score=prediction[1],
        success=True
    )
```

## Failure Conditions

The algorithm returns a failure (success=False) when:

1. Target word not found in sentence
2. No WordNet senses for the target word
3. Context extraction produces empty context
4. Embedding model unavailable or fails to encode
5. All candidate senses produce invalid vectors
6. All similarity calculations fail
7. Vector dimension mismatch between context and senses

In each case, the error is described in the `error_message` field.
