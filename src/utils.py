"""
utils.py
--------
Purpose: Shared utility functions used across the project - NLTK data path
setup, dataset loading from the NLTK Movie Reviews Corpus, saving/loading
of trained models with pickle, and simple helpers for reproducibility.
"""

import os
import random
import pickle

import nltk

# ---------------------------------------------------------------------
# Make sure NLTK can find the corpus data bundled inside data/nltk_data
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "nltk_data")
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED):
    """Fix random seed for reproducibility across the project."""
    random.seed(seed)


def load_movie_reviews():
    """
    Loads the NLTK Movie Reviews Corpus (2000 reviews, balanced
    1000 positive / 1000 negative) and returns two parallel lists:
    texts  -> raw review text (string)
    labels -> 'pos' or 'neg'
    """
    from nltk.corpus import movie_reviews

    documents = [
        (movie_reviews.raw(fileid), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]

    set_seed()
    random.shuffle(documents)

    texts = [doc for doc, _ in documents]
    labels = [label for _, label in documents]
    return texts, labels


def save_pickle(obj, path):
    """Save any Python object (model, vectorizer, etc.) to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """Load a previously pickled Python object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path):
    """Create a directory (and parents) if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)
    return path
