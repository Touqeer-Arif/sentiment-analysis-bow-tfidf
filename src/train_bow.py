"""
train_bow.py
------------
Purpose: Task 5 (Part 1) - trains a Logistic Regression sentiment
classifier using Bag-of-Words features, and saves the trained model so
it can be reloaded later without retraining (results/models/bow_model.pkl).
"""

import os
from sklearn.linear_model import LogisticRegression

from src.utils import save_pickle, RANDOM_SEED


def train_bow_model(X_train_bow, y_train):
    """
    Trains a Logistic Regression classifier on Bag-of-Words features.
    max_iter is raised because BoW/TF-IDF feature spaces are high
    dimensional and sparse, which the default 100 iterations of
    liblinear/lbfgs sometimes isn't enough to converge on.
    """
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train_bow, y_train)
    return model


def save_bow_model(model, path="results/models/bow_model.pkl"):
    save_pickle(model, path)
    return path
