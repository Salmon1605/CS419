# IR Retrieval Project (TF-IDF, BM25, LSA)

## Overview

This project implements classic Information Retrieval models (TF-IDF, BM25, LSA) on the Cranfield dataset. The focus is on text preprocessing, inverted index construction, and retrieval evaluation.

## Key Features

- Text preprocessing: lowercase, punctuation/numeric removal, stopwords, lemmatization, tokenization.
- Inverted index for candidate document filtering.
- TF-IDF and BM25 scoring.
- LSA (SVD on TF-IDF) with cosine similarity in latent space.
- Evaluation and experiments via notebooks in test/.

## Data

- Cranfield dataset: located under Dataset/ (queries, qrels, and DOCS files).
- CSV files in project/data/:
  - documents.csv: requires `id` and `content` columns.
  - queries.csv: query format used by the test notebooks.

## Setup

Recommended Python 3.9+ with these dependencies:

- pandas
- numpy
- nltk
- scikit-learn
- scipy
- matplotlib

Example install:

```bash
pip install pandas numpy nltk scikit-learn scipy matplotlib
```

Download required NLTK resources (stopwords, tokenizer, lemmatizer):

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
```

## Usage

### 1) Notebook workflow

Run experiments and evaluation using notebooks in test/:

- test_tfidf.ipynb
- test_bm25.ipynb
- LSA_test.ipynb
- term_test.ipynb
- build_term_docs.ipynb

### 2) Module usage (short example)

```python
import pandas as pd
from source.utils.textPreprocessing import TextPreprocessing
from source.utils.inverted_index import InvertedIndex
from source.utils.Tf_Idf_retriever import TfIDF

# Load data (id, content)
df = pd.read_csv("data/documents.csv")

tp = TextPreprocessing()
df_proc = tp.process_dataframe(df)

index = InvertedIndex()
index.build_inverted_index(df_proc)

retriever = TfIDF(df_proc, index)
results = retriever.search("example query", top_k=10)
print(results)
```

## Evaluation

Notebooks under test/ provide evaluation for each model. Open the notebook and run cells in order to reproduce results.

## Project Structure (abridged)

```
project/
  data/
  source/
    utils/
  test/
  visualizations/
```

## Dataset Note

Cranfield is a classic IR dataset used in research and teaching. Please review its usage terms if you plan to redistribute it.
