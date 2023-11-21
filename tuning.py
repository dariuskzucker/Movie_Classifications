
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

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import normalize
import csv

def quadTune(X_train, Y_train):
    
    C_range = np.linspace(24, 28, num=100)
    gamma_range = np.linspace(0, .5, num=100)

    grid = np.column_stack((C_range, gamma_range))

    return select_param_quadratic_gamma(X_train, Y_train, k=5, metric="accuracy", param_range=grid)


def linearTune(X, Y):
    C_ranges = [10**i for i in range(-3, 4)]

    clf = select_param_linear_challenge(X, Y, k=5, metric="accuracy", C_range=C_ranges, loss="hinge", penalty="l2", dual=True)
    return clf



def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """Search for hyperparameters from the given candidates of quadratic SVM
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of a quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """

    best_C_val, best_r_val = 0.0, 0.0
    best_performance = -1

    for C, r in param_range:
        # print("Trying C: " + str(C) + ", r: " + str(r))
        clf = SVC(kernel='poly', degree=2, C=C, coef0=r, gamma='auto')
        performance = cv_performance(clf, X, y, k, metric)
        # print("Performance: " + str(performance))
        if performance > best_performance:
            best_C_val = C
            best_r_val = r
            best_performance = performance

    # print()
    print("Best c: " + str(best_C_val))
    print("Best r: " + str(best_r_val))
    print("CV Score " + str(best_performance))

    return best_C_val, best_r_val

def select_param_quadratic_gamma(X, y, k=5, metric="accuracy", param_range=[]):

    best_C_val, best_gamma = 10000000, 1000000
    best_performance = -1
    best_clf = None

    
    for C, g in param_range:
        print("Testing C: " + str(C))
        print("Testing gamma: " + str(g))
        clf = SVC(C=C, kernel='rbf', gamma=g)

        performance = cv_performance(clf, X, y, k, metric)
    
        if performance > best_performance:
            best_C_val = C
            best_gamma = g
            best_performance = performance
            best_clf = clf
        print("Performance: " + str(performance))
    print("Best C: " + str(best_C_val))
    print("Best gamma: " + str(best_gamma))

    # print()
    print("Best c: " + str(best_C_val))
    print("Best gamma: " + str(best_gamma))
    print("CV Score " + str(best_performance))

    return best_clf


def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True):
    """Search for hyperparameters from the given candidates of linear SVM with
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1")
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    
    best_C = -1
    best_performance = -1

    for C in C_range:
        clf = LinearSVC(penalty=penalty, loss=loss, dual=dual, C=C, random_state=445)
        performance = cv_performance(clf, X, y, k, metric)
        if performance > best_performance:
            best_C = C
            best_performance = performance

    print("Best c: " + str(best_C))
    print("CV Score " + str(best_performance))
    return best_C


def select_class_weights(
    X, y, k=5, metric="accuracy", cw_range=[], loss="hinge", penalty="l2", dual=True):
    """Search for optimal class weights from the given candidates of linear SVM with
    best k-fold CV performance.
    """
    best_cw = -1
    best_performance = -1

    for cw in cw_range:
        clf = LinearSVC(C=0.01, loss="hinge", penalty="l2", dual=True, random_state=445, class_weight = cw)
        performance = cv_performance(clf, X, y, k, metric)
        if performance > best_performance:
            best_cw = cw
            best_performance = performance

    print("Best cw: " + str(best_cw))
    print("CV Score " + str(best_performance))
    return best_cw
