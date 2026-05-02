import os 
import sys 
import glob 
import logging

import math
import scipy 
import sklearn 
import numpy as np 
import pandas as pd 

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter 
from textPreprocessing import TextPreprocessing 

logging.basicConfig(format='%(asctime)s %(message)s')

class LSA: 
    def __init__(self, processed_docs, inverted_index_builder,vocab, dims:int=200):
        self.processed_docs = processed_docs
        self.inverted_index_builder = inverted_index_builder 
        self.inverted_index = self.inverted_index_builder.get_inverted_index() 
        self.vocab = vocab 
        self.dims = dims
        self.term_doc_matrix = None
        self.doc_ids = processed_docs["id"].tolist()
        self.text_preprocessing = TextPreprocessing()
        self.vectorizer = None
        self.vocab_index = {term: idx for idx, term in enumerate(self.vocab)}
        self.doc_embeddings = None
        self.doc_norms = None
        self.SVD = TruncatedSVD(n_components=self.dims)

    def _process_query(self, query): 
        terms = self.text_preprocessing._text_preprocessing(query) 
        return terms 
    
    def _build_terms_docs_matrix(self): 
        docs_size = self.processed_docs.shape[0]
        vocab_size = len(self.vocab)
        term_docs = np.zeros((vocab_size, docs_size), dtype=np.int32)

        for doc_idx in range(docs_size):
            token_counts = Counter(self.processed_docs['tokens'].iloc[doc_idx])
            for token, count in token_counts.items():
                term_idx = self.vocab_index.get(token)
                if term_idx is not None:
                    term_docs[term_idx, doc_idx] = count

        self.term_doc_matrix = term_docs

    def _build_terms_docs_matrix_tfidf(self):
        docs_tokens = self.processed_docs['tokens'].tolist()
        docs_text = [" ".join(tokens) for tokens in docs_tokens]
        self.vectorizer = TfidfVectorizer(vocabulary=self.vocab)
        tfidf_matrix = self.vectorizer.fit_transform(docs_text)
        self.term_doc_matrix = tfidf_matrix.T

    def _fit_svd(self): 
        if self.term_doc_matrix is None:
            self._vectorize()

        if scipy.sparse.issparse(self.term_doc_matrix):
            docs_terms = self.term_doc_matrix.T
        else:
            docs_terms = np.asarray(self.term_doc_matrix).T

        self.doc_embeddings = self.SVD.fit_transform(docs_terms)
        self.doc_norms = np.linalg.norm(self.doc_embeddings, axis=1)

    def _vectorize(self, use_tfidf: bool = True): 
        if use_tfidf:
            self._build_terms_docs_matrix_tfidf()
        else:
            self._build_terms_docs_matrix()

    def _get_embedding(self): 
        if self.doc_embeddings is None:
            self._fit_svd()
        return self.doc_embeddings

    def search(self, query, top_k:int=10): 
        processed_query = self._process_query(query)
        if not processed_query:
            return []

        if self.term_doc_matrix is None:
            self._vectorize()
        if self.doc_embeddings is None:
            self._fit_svd()

        if self.vectorizer is not None:
            query_text = " ".join(processed_query)
            query_vec = self.vectorizer.transform([query_text])
        else:
            query_vec = np.zeros((1, len(self.vocab)), dtype=np.float32)
            for token, count in Counter(processed_query).items():
                term_idx = self.vocab_index.get(token)
                if term_idx is not None:
                    query_vec[0, term_idx] = count

        query_emb = self.SVD.transform(query_vec)
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return []

        scores = (self.doc_embeddings @ query_emb.T).ravel().astype(np.float32)
        denom = self.doc_norms * query_norm
        scores = np.divide(scores, denom, out=np.zeros_like(scores), where=denom > 0)

        ranked = list(zip(self.doc_ids, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        if top_k is None:
            return ranked
        return ranked[:top_k]