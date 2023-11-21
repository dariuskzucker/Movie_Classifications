import itertools
import string
import warnings

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix
from feature_engineering import *
from preprocess import *
from tuning import *
from evaluate import *
from data_augment import *
from word_embeddings import *

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import normalize
import csv


import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import random


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)




def main():
    # Read binary data

    fname = "dataset.csv"
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname=fname
    )

    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname=fname
    )

    (
        X_train,
        Y_train,
        multiclass_dictionary,
    ) = get_multiclass_training_data()


    # X_test = (multiclass_dictionary)

    # print(X_train)

    # tester(X, Y, 0, 2)
    # tester(X, Y, 4, 2)
    tester(X, Y, 3, 2)
    # Y_pred = csf.predict(X_test)

    
def tester(X, Y, num_aug, n):
    preprocess(X)
    X_aug = augmentData(X, num_aug)
    Y_train = X_aug["label"].values.copy()
    X_train = csr_matrix(np.array(project1.featureEngineer(X_aug, n=n)))

    print("Linear Tuning on " + str(num_aug) + "x Augmented Data with n = " + str(n))
    linear_clf = linearTune(X_train, Y_train)

    print("Quad Tuning on " + str(num_aug) + "x Augmented Data with n = " + str(n))
    quad_clf = quadTune(X_train, Y_train)

    X_og = csr_matrix(np.array(project1.featureEngineer(X, n=n)))

    print("Pure linear perfomance on original training data")
    print(cv_performance(linear_clf, X_train, Y_train, k=5, metric="accuracy"))
    

    print("Pure quadratic perfomance on original training data")
    print(cv_performance(quad_clf, X_train, Y_train, k=5, metric="accuracy"))


if __name__ == "__main__":
    main()
