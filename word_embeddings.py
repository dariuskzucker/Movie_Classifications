
import itertools
import string
import warnings

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from helper import *
from matplotlib import pyplot as plt



warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)

def train_word2vec(fname):
    """
    Train a Word2Vec model using the Gensim library.
    First, iterate through all reviews in the dataframe, run your extract_word() function on each review, and append the result to the sentences list.
    Next, instantiate an instance of the Word2Vec class, using your sentences list as a parameter.
    Return the Word2Vec model you created.
    """
    df = load_data(fname)
    text_col = df['reviewText']

    sentences = []
    for index, row in enumerate(text_col):
        row = text_col[index]
        words = extract_word(row)
        sentences.append(words)

    return Word2Vec(sentences=sentences, workers=1)


def compute_association(fname, w, A, B):
    """
    Inputs:
        - fname: name of the dataset csv
        - w: a word represented as a string
        - A and B: sets that each contain one or more English words represented as strings
    Output: Return the association between w, A, and B as defined in the spec
    """
    model = train_word2vec(fname)
    w_embed = model.wv[w]

    # First, we need to find a numerical representation for the English language words in A and B
    # TODO: Complete words_to_array(), which returns a 2D Numpy Array where the ith row is the embedding vector for the ith word in the input set.
    def words_to_array(set):
        l = []
        for word in set:
            l.append(model.wv[word])
        return np.array(l)

    # TODO: Complete cosine_similarity(), which returns a 1D Numpy Array where the ith element is the cosine similarity
    #      between the word embedding for w and the ith embedding in the array representation of the input set
    def cosine_similarity(set):
        array = words_to_array(set)
       
        dot = np.dot(array, w_embed)

        norm_array = np.linalg.norm(array, axis=1)
        norm_w = np.linalg.norm(w_embed)
        return dot / (norm_array * norm_w)

    # TODO: Return the association between w, A, and B.
    #      Compute this by finding the difference between the mean cosine similarity between w and the words in A, and the mean cosine similarity between w and the words in B
    return np.mean(cosine_similarity(A)) - np.mean(cosine_similarity(B))


    model = train_word2vec(fname)
    close_words = model.wv.most_similar('plot', topn=5)
    for word in close_words:
        print(word)