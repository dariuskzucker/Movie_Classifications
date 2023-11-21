
import itertools
import string

import numpy as np
import pandas as pd
from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from scipy.sparse import csr_matrix

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import random

def extract_word(input_string):
    # Preprocess review text into list of tokens.

    # Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    # Return the resulting array.


    input_string = input_string.lower()
    for p in string.punctuation:
        input_string = input_string.replace(p, " ")
    return input_string.split()


def extract_dictionary(df):
    # Map words to index.

    # Reads a pandas dataframe, and returns a dictionary of distinct words
    # mapping from each distinct word to its index (ordered by when it was
    # found).
    word_dict = {}
    concat = df['reviewText'].str.cat(sep=' ')
    words = extract_word(concat)

    i=0
    for w in words:
        if w not in word_dict:
            word_dict[w] = i
            i+=1
    
    return word_dict


def generate_feature_matrix(df, word_dict):
    # Create matrix of feature vectors for dataset.

    # Reads a dataframe and the dictionary of unique words to generate a matrix
    # of {1, 0} feature vectors for each review. For each review, extract a token
    # list and use word_dict to find the index for each token in the token list.
    # If the token is in the dictionary, set the corresponding index in the review's
    # feature vector to 1. The resulting feature matrix should be of dimension
    # (# of reviews, # of words in dictionary).
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    

    text_col = df['reviewText']
    for index, row in enumerate(text_col):
        row = text_col[index]
        words = extract_word(row)
        for i,w in enumerate(word_dict):
            if w in words:
                feature_matrix[index][i] = 1

    return feature_matrix


def featureEngineer(df, n):
    # Use Bag of NGrams model with POS Tagging
    unique_ngrams = generate_NGrams(df, n)
    NGram_matrix = generate_ngram_matrix(df, unique_ngrams)
    normalized_ngrams = normalizer(NGram_matrix)
    return normalized_ngrams

def normalizer(X):
    # Normalize feature matrix
    return normalize(X, norm='l2', axis=1)

def generate_ngram_matrix(df, unique_ngrams):
    # Creates a feature matrix for bag of ngrams

    number_of_reviews = df.shape[0]
    number_of_ngrams= len(unique_ngrams)
    feature_matrix = np.zeros((number_of_reviews, number_of_ngrams))

    for index, words in enumerate(df['word_list']):
        print(index)
        for i, ngram in enumerate(unique_ngrams):
                feature_matrix[index][i] = countNGrams(words, ngram)

    return feature_matrix

def generate_NGrams(df, k):
    # Generates a list of unique ngrams where n is [0, k]

    unique_ngrams = []
    for i in range(1, k+1):

        for index, words in enumerate(df['word_list']):
            line_ngram = createNGram(words, i)
            
            for ngram in line_ngram:
                if ngram not in unique_ngrams:
                    unique_ngrams.append(ngram)

    return unique_ngrams

def createNGram(words, k):
    # Create a list of unique ngrams (tuples),
    # based on a list of words where n=k
    ngrams = []
    for i in range(len(words)-k+1):
        ngram = tuple(words[i:i+k])
        if ngram not in ngrams:
            ngrams.append(ngram)
    return ngrams

def countNGrams(words, ngram):
    # Counts # of ngram appearances in list of words
    count = len(ngram)
    total = 0
    for i in range(len(words)-count+1):
        found = True
        for j in range(count):
            if (words[i+j]) != ngram[j]:
                found = False
                break
        if found:
            total += 1
    return total