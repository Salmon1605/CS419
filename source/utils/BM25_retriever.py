import os 
import sys 
import glob 
import logging

import math
import numpy as np 
import pandas as pd 

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter 
from textPreprocessing import TextPreprocessing 

logging.basicConfig(format='%(asctime)s %(message)s')

class BM25Retriever: 
    def __init__(self, processed_docs, inverted_index_builder, k1:float=1.5, b:float=0.75): 
        # CORE
        self.processed_docs = processed_docs 
        self.inverted_index_builder = inverted_index_builder 
        self.text_preprocessing = TextPreprocessing()

        # HELPERS
        self.inverted_index = self.inverted_index_builder.get_inverted_index()
        self.total_documents = self.inverted_index_builder.get_total_documents() 
        self.document_lengths = self.inverted_index_builder.get_documents_length()
        self.average_document_length = self.inverted_index_builder.get_average_docs_length()

        # HYPERPARAMETERS
        self.k1 = k1  
        self.b = b

    def _process_query(self, query): 
        terms = self.text_preprocessing._text_preprocessing(query) 
        return terms

    def _get_candidates_docs(self, terms): 
        candidates = self.inverted_index_builder.get_candidate_documents(terms) 
        return candidates 

    def _calculate_idf(self, term): 
        if term in self.inverted_index: 
            term_document_frequency = self.inverted_index[term]['df'] 
            term_idf = np.log((self.total_documents - term_document_frequency + 0.5) / (term_document_frequency + 0.5)) 
            return term_idf     
        else: 
            return 0 

    def _score_term(self, term, doc_id): 
        if term not in self.inverted_index:
            return 0.0

        postings = self.inverted_index[term]['postings']
        tf = postings.get(doc_id, 0)
        if tf == 0:
            return 0.0

        idf = self._calculate_idf(term=term)
        doc_len = self.document_lengths.get(doc_id, 0)
        avgdl = self.average_document_length
        if avgdl == 0:
            return 0.0

        denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / avgdl))
        if denom == 0:
            return 0.0

        return idf * (tf * (self.k1 + 1)) / denom

    def _score_document(self, doc_id, query_terms): 
        if not query_terms:
            return 0.0

        score = 0.0
        for term in query_terms:
            score += self._score_term(term, doc_id)

        return score

    def search(self, query, top_k:int=10): 
        processed_query = self._process_query(query)
        if not processed_query:
            return []

        candidate_docs = self._get_candidates_docs(processed_query)
        if not candidate_docs:
            return []

        scores = []
        for doc_id in candidate_docs:
            score = self._score_document(doc_id, processed_query)
            if score != 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        if top_k is None:
            return scores
        return scores[:top_k]