import os 
import sys 
import glob 

import math
import numpy as np 
import pandas as pd 

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter 

class InvertedIndex:
    def __init__(self, processed_docs_df: pd.DataFrame, vocab: list):
        self.processed_docs_df = processed_docs_df 
        self.vocab = vocab 
        self.inverted_index = {}
        self.document_lengths = {} 
        self.total_documents = len(self.processed_docs_df)  
        self.average_document_length = 0 

    def _build(self) -> None: 
        """ 
        Iterate through processed document to get doc_id and term of that docs 
    
        Input: processed_df 
                - doc_id 
                - content 
                - tokens 

        Output: { 
                    term: { "df": ..., "postings": { doc_id: tf } } 
                    
                }
                - df: docs_frequency 
                - doc_id 
                - tf: term_frequency in that doc
        """ 
        
        for row in self.processed_docs_df.itertuples(): 
            doc_id = row.id 
            # doc_content = row.content 
            doc_terms = row.tokens 
            self.document_lengths[doc_id] = len(doc_terms)

            term_counter = Counter(doc_terms)
            for term, tf in term_counter.items(): 
                if term in self.inverted_index:
                    self.inverted_index[term]['df'] += 1
                    self.inverted_index[term]['postings'][doc_id] = tf


                else: 
                    self.inverted_index[term] = {
                        "df":0, 
                        "postings": {}
                    }
                    self.inverted_index[term]['df'] += 1
                    self.inverted_index[term]['postings'][doc_id] = tf

            # Calculating average documents length
            sum_docs_length = sum([v for _,v in self.document_lengths.items()])
            total_docs = self.total_documents 
            if total_docs == 0:
                self.average_document_length = 0 
            else: 
                avgdl = sum_docs_length/total_docs 
                self.average_document_length = avgdl

    def get_total_documents(self): 
        return self.total_documents

    def get_inverted_index(self): 
        return self.inverted_index
                    
    def get_average_docs_length(self): 
        return self.average_document_length
    
    def get_postings_by_terms(self, terms): 
        postings = [] 
        for term in terms: 
            posting = self.inverted_index[term]['postings']
            postings.append({term:posting}) 

        return postings 
    
    def get_documents_frequency(self, term): 
        df = self.inverted_index[term]['df'] 
        return df
    
    def get_documents_length(self): 
        return self.document_lengths
    
    def get_candidate_documents(self, terms, option:str='union'): 
        candidates = set()
        if option == 'union':
            for term in terms:
                if term not in self.inverted_index:
                    continue
                docs_id = self.inverted_index[term]['postings'].keys()
                candidates.update(docs_id)

            return sorted(list(candidates))

        elif option == 'intersection':
            if not terms:
                return []
            candidates = None
            for term in terms:
                if term not in self.inverted_index:
                    return []
                docs_id = set(self.inverted_index[term]['postings'].keys())
                if candidates is None:
                    candidates = docs_id
                else:
                    candidates &= docs_id

            return sorted(list(candidates))
        else:
            raise ValueError("option must be 'union' or 'intersection'")
              
