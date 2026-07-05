"""
evaluation.py
-------------
Purpose: Task 6 (Model Evaluation) and Task 7 (Feature Importance) -
computes Accuracy / Precision / Recall / F1 / Confusion Matrix for a
trained model, builds a side-by-side comparison table for BoW vs
TF-IDF, and extracts the most positive/negative words learned by the
TF-IDF Logistic Regression model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test, pos_label="pos"):
    """
    Computes the five evaluation metrics required by Task 6.
    Returns a dict of metrics plus the raw confusion matrix.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=pos_label),
        "recall": recall_score(y_test, y_pred, pos_label=pos_label),
        "f1_score": f1_score(y_test, y_pred, pos_label=pos_label),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["neg", "pos"]),
        "y_pred": y_pred,
    }
    return metrics


def build_comparison_table(bow_metrics, tfidf_metrics):
    """
    Produces the Metric | BoW | TF-IDF comparison table exactly as
    requested in Task 6's example table.
    """
    data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "BoW": [
            bow_metrics["accuracy"],
            bow_metrics["precision"],
            bow_metrics["recall"],
            bow_metrics["f1_score"],
        ],
        "TF-IDF": [
            tfidf_metrics["accuracy"],
            tfidf_metrics["precision"],
            tfidf_metrics["recall"],
            tfidf_metrics["f1_score"],
        ],
    }
    return pd.DataFrame(data)


def top_n_words(model, vectorizer, n=10):
    """
    Task 7: For a Logistic Regression model trained on TF-IDF features,
    extracts the top-N words pushing predictions towards 'pos' (largest
    positive coefficients) and towards 'neg' (largest negative
    coefficients).

    LogisticRegression's classes_ are alphabetically ordered ['neg','pos'],
    so a positive coefficient increases the log-odds of the 'pos' class.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    top_pos_idx = np.argsort(coefs)[-n:][::-1]
    top_neg_idx = np.argsort(coefs)[:n]

    top_positive = list(zip(feature_names[top_pos_idx], coefs[top_pos_idx]))
    top_negative = list(zip(feature_names[top_neg_idx], coefs[top_neg_idx]))
    return top_positive, top_negative
