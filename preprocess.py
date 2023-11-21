
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# Apply all of the preprocessing functions
#   1. Extract text into list of words
#   2. Delete stopwords
#   3. Lemmatize list of words
# Save to disk for future use
def preprocess(df):
    convert_text_to_word_list(df)
    delete_stopwords(df)
    lemmatize(df)

# Add column 'word_list' that applies extract_word
# -> lowercase list of words without punctuation
def convert_text_to_word_list(df):
    df['word_list'] = df['reviewText'].apply(extract_word)

# Delete stopwords from 'word_list' column
def delete_stopwords(df):
    stop_words = set(stopwords.words("english"))
    df['word_list'] = df['word_list'].apply(lambda row: [word for word in row if word.lower() not in stop_words])

# Apply lemmatization to 'word_list' column
def lemmatize(df):
    wn = nltk.WordNetLemmatizer()
    df['word_list'] = df['word_list'].apply(lambda row: [wn.lemmatize(word) for word in row])