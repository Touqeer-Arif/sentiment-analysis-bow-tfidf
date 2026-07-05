"""
train_tfidf.py
--------------
Purpose: Task 5 (Part 2) - trains a Logistic Regression sentiment
classifier using TF-IDF features, and saves the trained model so it can
be reloaded later without retraining (results/models/tfidf_model.pkl).
"""

from sklearn.linear_model import LogisticRegression

from src.utils import save_pickle, RANDOM_SEED


def train_tfidf_model(X_train_tfidf, y_train):
    """
    Trains a Logistic Regression classifier on TF-IDF features.
    Same hyperparameters as the BoW model so that any performance
    difference reflects the feature representation, not the classifier
    configuration.
    """
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train_tfidf, y_train)
    return model


def save_tfidf_model(model, path="results/models/tfidf_model.pkl"):
    save_pickle(model, path)
    return path
