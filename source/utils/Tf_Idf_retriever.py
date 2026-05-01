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

class TfIDF: 
    def __init__(self, inverted_index, processed_docs: pd.DataFrame):
        self.inverted_index_builder = inverted_index 
        self.processed_docs = processed_docs
        self.total_documents = self.inverted_index_builder.get_total_documents()
        self.text_preprocessing = TextPreprocessing()
        self.inverted_index = self.inverted_index_builder.get_inverted_index()

    def _process_query(self, query: str): 
        terms = self.text_preprocessing._text_preprocessing(query) 
        return terms

    def _get_candidate_docs(self, terms): 
        candidates = self.inverted_index_builder.get_candidate_documents(terms) 
        return candidates 

    def _calculate_idf(self, term: str): 
        if term in self.inverted_index: 
            term_document_frequency = self.inverted_index[term]['df'] 
            term_idf = np.log(self.total_documents / term_document_frequency) 
            return term_idf     
        else: 
            return 0
        
    def _build_query_vector(self, processed_query):
        query_tf_idf = {} 
        terms_counter = Counter(processed_query)
        for term, tf in terms_counter.items(): 
            term_tf_idf =  tf * self._calculate_idf(term) 
            query_tf_idf.update({term:term_tf_idf})
        return query_tf_idf
    

    def search(self, query:str, top_k:int = 10): 
        processed_query = self._process_query(query)
        if not processed_query:
            return []

        query_vector = self._build_query_vector(processed_query)
        query_norm = math.sqrt(sum(v * v for v in query_vector.values()))
        if query_norm == 0:
            return []

        candidate_docs = self._get_candidate_docs(processed_query)
        doc_tokens_map = self.processed_docs.set_index("id")["tokens"].to_dict()

        scores = []
        for doc_id in candidate_docs:
            tokens = doc_tokens_map.get(doc_id, [])
            if not tokens:
                continue

            term_counter = Counter(tokens)
            doc_tf_idf = {}
            for term, tf in term_counter.items():
                doc_tf_idf[term] = tf * self._calculate_idf(term)

            doc_norm = math.sqrt(sum(v * v for v in doc_tf_idf.values()))
            if doc_norm == 0:
                continue

            dot_product = 0.0
            for term, q_weight in query_vector.items():
                dot_product += q_weight * doc_tf_idf.get(term, 0.0)

            score = dot_product / (doc_norm * query_norm)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        if top_k is None:
            return scores
        return scores[:top_k]