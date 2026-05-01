import os 
import sys 
import glob 
import re
import nltk 
import logging 

import numpy as np 
import pandas as pd 

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter


class BooleanModel: 
    def __init__(self):
        pass