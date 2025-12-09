# Legal Contract Information Extraction

## 📄 Project Overview

This project addresses the challenge of automating data extraction from **unstructured legal and procurement documents (specifically in Russian)**.

The core objective is to implement an **Extractive Question Answering (QA)** pipeline capable of accurately locating and extracting critical contract clauses and specific data points (e.g., "Contract Security," "Warranty Obligations") from long-form text documents.

This repository demonstrates the development of a robust baseline model and outlines the strategy for fine-tuning a state-of-the-art model for production-grade performance.

## 🛠 Technical Approach

The extraction process is structured in two distinct phases: a semantic similarity baseline and a specialized span-extraction fine-tuning phase.

### Phase 1: Semantic Similarity Baseline (Current Implementation)

The initial implementation uses a computationally efficient **semantic similarity** approach to quickly and accurately identify the most relevant sentence or clause within a document based on a target query (the `label`).

* **Goal:** To find the sentence within the document (`text`) that is most semantically similar to the required information (`label`).
* **Methodology:**
    1.  The document is tokenized and segmented into individual sentences.
    2.  Sentence Embeddings are generated for every sentence in the document and for the target query/label.
    3.  **Cosine Similarity** is calculated between the query embedding and every sentence embedding.
    4.  The sentence with the highest similarity score is returned as the predicted answer span.
* **Core Model:** **`distiluse-base-multilingual-cased-v1`** (Sentence-BERT architecture).

### Phase 2: Model Fine-Tuning

To achieve higher precision and handle long-context documents typical in legal text, the project will advance to a dedicated span-extraction architecture.

* **Goal:** Predict the precise start and end character indices of the answer span, rather than relying on full sentence boundaries.
* **Methodology:** Fine-tuning the Sentence-BERT model on the provided dataset (`id`, `text`, `label`, `extracted_part_answer_start`, `extracted_part_answer_end`) to perform the span-prediction task directly.

## 📦 Key Technologies

* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `torch`
* **NLP Framework:** `Hugging Face Transformers` / `Sentence Transformers`
* **Model:** `distiluse-base-multilingual-cased-v1`

Model used: distiluse-base-multilingual-cased-v1
