"""
feature_extraction.py
----------------------
Purpose: Implements Task 4 - converting cleaned text into numerical
feature representations using:
    Part A: Bag of Words (CountVectorizer)
    Part B: TF-IDF (TfidfVectorizer)

Both extractors are capped at a shared max_features so that the two
representations are directly comparable in Task 5/6, and both use the
same n-gram range (unigrams) for a fair, apples-to-apples comparison.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

MAX_FEATURES = 5000


def build_bow_features(train_texts, test_texts, max_features=MAX_FEATURES):
    """
    Fits a Bag-of-Words CountVectorizer on the training set only (to
    avoid data leakage) and transforms both train and test sets.
    Returns: vectorizer, X_train_bow, X_test_bow
    """
    vectorizer = CountVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test


def build_tfidf_features(train_texts, test_texts, max_features=MAX_FEATURES):
    """
    Fits a TF-IDF vectorizer on the training set only and transforms
    both train and test sets.
    Returns: vectorizer, X_train_tfidf, X_test_tfidf
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test


def vocabulary_summary(vectorizer, X, name="Vectorizer"):
    """
    Returns a small dict summarising a fitted vectorizer: vocabulary
    size, a sample of vocabulary terms, and the feature matrix shape -
    exactly what Task 4 asks to be displayed.
    """
    vocab = vectorizer.get_feature_names_out()
    return {
        "name": name,
        "vocabulary_size": len(vocab),
        "sample_vocabulary": list(vocab[:20]),
        "feature_matrix_shape": X.shape,
    }
