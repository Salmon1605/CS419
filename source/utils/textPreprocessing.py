import os 
import sys 
import re 
import glob 
import nltk 
import logging 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logging.basicConfig(format='%(asctime)s %(message)s')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

class TextPreprocessing:
    def __init__(self):
        # self.dataframe = dataframe.copy()
        self.temp_dataframe = None

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    # ---------------------------
    # BASIC CLEANING
    # ---------------------------
    def _basic_process(self, text: str):
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ---------------------------
    # TOKENIZATION
    # ---------------------------
    def _tokenize(self, text: str):
        return word_tokenize(text)

    # ---------------------------
    # REMOVE NUMERIC TOKENS
    # ---------------------------
    def _remove_numeric_tokens(self, tokens):
        return [t for t in tokens if not re.search(r"\d", t)]

    # ---------------------------
    # REMOVE STOPWORDS
    # ---------------------------
    def _remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words]

    # ---------------------------
    # LEMMATIZATION
    # ---------------------------
    def _lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    # ---------------------------
    # FULL PIPELINE
    # ---------------------------
    def _text_preprocessing(self, text: str):
        text = self._basic_process(text)
        tokens = self._tokenize(text)

        # remove numbers FIRST (important design choice)
        tokens = self._remove_numeric_tokens(tokens)

        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)

        return tokens

    # ---------------------------
    # APPLY TO DATAFRAME
    # ---------------------------
    def process_dataframe(self, dataframe:pd.DataFrame):
        temp_df = dataframe.copy()

        temp_df["content"] = temp_df["content"].fillna("").astype(str)
        temp_df["tokens"] = temp_df["content"].apply(self._text_preprocessing).apply(tuple)

        self.temp_dataframe = temp_df
        return temp_df
    

    # ---------------------------
    # BUILD VOCAB
    # ---------------------------
    def get_vocab(self):
        vocab = set()
        for tokens in self.temp_dataframe["tokens"]:
            vocab.update(tokens)

        return sorted(vocab)
