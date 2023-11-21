

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

def performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
        
    if metric == 'f1-score':
        return metrics.f1_score(y_true, y_pred)
    elif metric == 'auroc':
        return metrics.roc_auc_score(y_true, y_pred)
    elif metric == 'precision':
        return metrics.precision_score(y_true, y_pred)
    elif metric == 'sensitivity':
        return metrics.recall_score(y_true, y_pred)
    elif metric == 'specificity':
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[-1,1]).ravel()
        return tn / (tn + fp)
    else:
        return metrics.accuracy_score(y_true, y_pred)


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """

    # Put the performance of the model on each fold in the scores array
    scores = []

    skf = StratifiedKFold(n_splits=k)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)): 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        predictor = clf.fit(X_train, y_train)

        if (metric == 'auroc'):
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)

        score = performance(y_test, y_pred, metric)
        scores.append(score)

    return np.array(scores).mean()



def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []

    for C in C_range:
        clf = LinearSVC(penalty=penalty, loss=loss, dual=dual, C=C, random_state=445)
        predictor = clf.fit(X, y)
        theta = clf.coef_[0]

        norm0.append(sum(1 for w in theta if w != 0))


    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.show()
    plt.close()


