"""The About & Learn page: reference material for the method."""

from __future__ import annotations

import streamlit as st


def render_about() -> None:
    """Render the reference and teaching material."""
    st.title("📚 About & Learn")
    st.markdown(
        "Educational resources for understanding Word Sense Disambiguation "
        "and how ContextSense AI works."
    )

    about_tabs = st.tabs(
        [
            "What is WSD?",
            "How it Works",
            "WordNet & Synsets",
            "Embeddings",
            "Evaluation Metrics",
            "Limitations & Future Work",
        ]
    )

    with about_tabs[0]:
        st.markdown(
            """
            ## What is Word Sense Disambiguation?

            **Word Sense Disambiguation (WSD)** is the task of determining which
            meaning of an ambiguous word is being used in a particular context.

            ### Example

            The word **"bank"** has multiple meanings:

            | Context | Intended Meaning |
            |---------|-----------------|
            | "I deposited money in the **bank**." | Financial institution |
            | "The fisherman sat near the **bank** of the river." | Land beside water |
            | "The plane flew over the **bank**." | Cloud bank / snow bank |

            Humans disambiguate effortlessly using context. Computers need
            explicit methods to do the same.

            ### Why is WSD Hard?

            - Many words have dozens of possible senses
            - Contexts can be ambiguous themselves
            - Some senses are rare and poorly represented in data
            - Related senses can be very close in meaning
            """
        )

    with about_tabs[1]:
        st.markdown(
            """
            ## How ContextSense AI Works

            Our pipeline has 5 stages:

            ### 1. Candidate Generation (WordNet)
            For the target word, retrieve all possible meanings (synsets) from WordNet.

            ### 2. Context Extraction
            Extract surrounding words as context, excluding the target word itself.

            ### 3. Vector Representation
            - **Context vector**: Encode context words into a single vector
            - **Sense vectors**: Encode each candidate sense description into a vector

            ### 4. Similarity Calculation
            Compute cosine similarity between context vector and each sense vector.

            ### 5. Ranking and Prediction
            Rank candidates by similarity score; select the highest-ranked sense.

            ### The Cosine Similarity Formula

            ```
            cosine_similarity(A, B) = dot(A, B) / (norm(A) × norm(B))
            ```

            - Range: [-1, 1]
            - 1 = identical direction
            - 0 = orthogonal (no similarity)
            - -1 = opposite direction

            **Note:** Similarity is NOT probability. A score of 0.8 does not
            mean "80% confident this is correct."
            """
        )
        st.divider()
        st.subheader("Architecture Diagram")

        # Streamlit has no built-in Mermaid renderer, so a ```mermaid fence
        # displayed as raw source. Plain text renders everywhere.
        st.code(
            """
  sentence
      |
      v
  tokenize  ->  locate target occurrence (POS tag here too)
      |                        |
      |                        v
      |               +-------------------+
      |               | WordNet candidates|-- POS filter
      |               +-------------------+
      |                        |
      |                        v
      |                 candidate senses
      |                        |
      |        definition + examples + lemmas + hypernyms
      |                        |
      |                        v
      |                 sense description ------------+
      |                                                  |
      v                                                  v
  context tokens  (target removed, window or full) --> embedding --> cosine
                                                             --> rank
                                                             --> top sense
            """.strip(),
            language="text",
        )

    with about_tabs[2]:
        st.markdown(
            """
            ## WordNet and Synsets

            **WordNet** is a lexical database that groups English words into
            sets of synonyms called **synsets**.

            ### What is a Synset?

            A synset is a set of synonyms that represent a single distinct concept.

            Example: The synset **bank.n.02** contains:
            - bank
            - banking_concern
            - banking_company
            - depository_financial_institution

            Its gloss is "a financial institution that holds money deposits".

            ### Synset ID Format

            WordNet sense identifiers use the format: `lemma.pos.sense_number`

            - **lemma**: A representative word for the sense
            - **pos**: Part of speech (n=noun, v=verb, a=adjective, r=adverb)
            - **sense_number**: That sense's position among the lemma's senses,
              counting from 01 in the order WordNet stores them

            The numbering is per-lemma, so the financial sense of "bank" is
            `bank.n.02`, not `bank.n.01`. WordNet stores the landform sense
            ("sloping land beside a body of water") first.

            ### Semantic Relations

            WordNet encodes relationships between synsets:

            - **Hypernyms**: More general concepts ("vehicle" is a hypernym of "car")
            - **Hyponyms**: More specific concepts ("car" is a hyponym of "vehicle")
            - **Meronyms**: Part-of relationships ("wheel" is a meronym of "car")
            - **Holonyms**: Has-part relationships ("car" is a holonym of "wheel")
            """
        )

    with about_tabs[3]:
        st.markdown(
            """
            ## Word Embeddings

            **Embeddings** are dense numerical vectors that represent words or
            texts in a continuous vector space where similar meanings are close together.

            ### One-Hot vs. Dense Embeddings

            | One-Hot Encoding | Dense Embeddings |
            |------------------|------------------|
            | Sparse, high-dimensional | Dense, lower-dimensional |
            | Each word is orthogonal | Similar words have similar vectors |
            | No notion of similarity | Captures semantic relationships |
            | Size = vocabulary size | Size = embedding dimension |

            ### Word2Vec

            Word2Vec (Mikolov et al., 2013) learns word embeddings by predicting
            a word from its context (CBOW) or predicting context from a word (Skip-gram).

            - **CBOW**: Predict target word from surrounding context
            - **Skip-gram**: Predict surrounding words from target word

            Word2Vec produces **static** embeddings: the same word always gets the
            same vector regardless of context.

            ### Sentence Transformers

            Sentence Transformers (Reimers & Gurevych, 2019) extend transformer
            models to produce embeddings for full sentences or passages.

            - Based on models like BERT, RoBERTa
            - Produces **contextual** embeddings
            - Better at capturing sentence-level semantics

            ### Mean Pooling

            When we have multiple word vectors (e.g., for a sentence), we can
            combine them with **mean pooling**: average all vectors to get one vector.

            ```
            context_vector = (vec_1 + vec_2 + ... + vec_n) / n
            ```

            **Limitations of mean pooling:**
            - Loses word order
            - Treats all words equally
            - Cannot model complex interactions
            """
        )

    with about_tabs[4]:
        st.markdown(
            """
            ## Evaluation Metrics

            ### Accuracy
            The fraction of correct predictions:
            ```
            accuracy = correct_predictions / total_predictions
            ```

            ### Precision, Recall, F1

            For each sense/class:

            - **Precision**: Of all times we predicted this sense, how often were we right?
            - **Recall**: Of all times this sense was correct, how often did we predict it?
            - **F1**: Harmonic mean of precision and recall

            ### Macro vs. Weighted Averaging

            - **Macro**: Average metrics across all classes, treating each equally
            - **Weighted**: Average weighted by class frequency

            ### Why Multiple Metrics?

            Accuracy alone can be misleading:
            - A model that always picks the most frequent sense may have high accuracy
            - But it fails on less common senses
            - Per-class metrics reveal these weaknesses

            ### What We Report

            - **Accuracy**: Overall correctness
            - **Precision (macro)**: Average precision across senses
            - **Recall (macro)**: Average recall across senses
            - **F1 (macro)**: Balanced measure across all senses
            - **F1 (weighted)**: F1 weighted by sense frequency
            - **Per-class metrics**: Performance for each individual sense
            """
        )

    with about_tabs[5]:
        st.markdown(
            """
            ## Limitations

            ### 1. Context Window Limitations
            Our local window context extraction captures nearby words but misses:
            - Long-range dependencies
            - Syntactic relationships
            - Discourse-level context

            ### 2. Sense Description Quality
            WordNet definitions may not be sufficient to distinguish closely related senses.
            Some senses lack example sentences.

            ### 3. Embedding Model Limitations
            - **Word2Vec**: Static embeddings, no context sensitivity,
              mean pooling loses information
            - **Sentence Transformers**: Better but still may not capture
              fine-grained sense distinctions

            ### 4. Evaluation Dataset
            The bundled 8-example sample is for demonstration only. The SemCor
            conversion and evaluation code ship here; the corpus itself is
            fetched on demand. Run `uv run run-evaluation` to reproduce, and see
            the committed results in `docs/evaluation.md`.

            ### 5. Baseline Limitations
            - First-sense baseline doesn't use context
            - Most-frequent-sense baseline uses approximate frequency data

            ## Future Improvements

            1. **Better context extraction**: Use dependency parsing to find
              semantically relevant context words
            2. **Sense embeddings**: Train dedicated sense embeddings rather than using descriptions
            3. **Fine-tuned transformers**: Fine-tune a transformer on WSD tasks
            4. **Ensemble methods**: Combine multiple approaches
            5. **Wider benchmarks**: Evaluate on SemEval or other WSD benchmarks
            6. **Domain adaptation**: Adjust for specific domains (medical, legal, etc.)
            """
        )

    st.divider()
    st.subheader("References")
    st.markdown(
        """
        - Miller, G. A. (1995). WordNet: A lexical database for English.
          *Communications of the ACM*, 38(11), 39-41.
        - Mikolov, T., et al. (2013). Efficient estimation of word representations
          in vector space. *arXiv:1301.3781*.
        - Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings
          using Siamese BERT-networks. *EMNLP-IJCNLP 2019*.
        - Navigli, R. (2009). Word sense disambiguation: A survey.
          *ACM Computing Surveys*, 41(2), 1-69.
        """
    )
