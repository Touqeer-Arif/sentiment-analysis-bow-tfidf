"""
main.py
-------
Purpose: Standalone command-line entry point that runs the entire
sentiment analysis pipeline (data loading -> preprocessing -> feature
extraction -> training -> evaluation) end to end using only the `src/`
modules, without needing Jupyter. Useful for quick re-runs / viva demos.

Usage:
    python main.py
"""

from sklearn.model_selection import train_test_split

from src.utils import load_movie_reviews, set_seed, ensure_dir
from src.preprocessing import preprocess_pipeline
from src.feature_extraction import build_bow_features, build_tfidf_features, vocabulary_summary
from src.train_bow import train_bow_model, save_bow_model
from src.train_tfidf import train_tfidf_model, save_tfidf_model
from src.evaluation import evaluate_model, build_comparison_table, top_n_words


def main():
    set_seed()
    ensure_dir("results/plots")
    ensure_dir("results/models")
    ensure_dir("results/tables")

    print("Loading NLTK Movie Reviews Corpus...")
    texts, labels = load_movie_reviews()
    print(f"  Total: {len(texts)}  Positive: {labels.count('pos')}  Negative: {labels.count('neg')}")

    print("Running preprocessing pipeline on all reviews...")
    cleaned_texts = [preprocess_pipeline(t) for t in texts]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        cleaned_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Building Bag-of-Words features...")
    bow_vectorizer, X_train_bow, X_test_bow = build_bow_features(X_train_text, X_test_text)
    print(" ", vocabulary_summary(bow_vectorizer, X_train_bow, "BoW"))

    print("Building TF-IDF features...")
    tfidf_vectorizer, X_train_tfidf, X_test_tfidf = build_tfidf_features(X_train_text, X_test_text)
    print(" ", vocabulary_summary(tfidf_vectorizer, X_train_tfidf, "TF-IDF"))

    print("Training Logistic Regression on BoW features...")
    bow_model = train_bow_model(X_train_bow, y_train)
    save_bow_model(bow_model, "results/models/bow_model.pkl")

    print("Training Logistic Regression on TF-IDF features...")
    tfidf_model = train_tfidf_model(X_train_tfidf, y_train)
    save_tfidf_model(tfidf_model, "results/models/tfidf_model.pkl")

    print("Evaluating both models...")
    bow_metrics = evaluate_model(bow_model, X_test_bow, y_test)
    tfidf_metrics = evaluate_model(tfidf_model, X_test_tfidf, y_test)

    table = build_comparison_table(bow_metrics, tfidf_metrics)
    table.to_csv("results/tables/comparison_table.csv", index=False)
    print(table.to_string(index=False))

    print("\nTop TF-IDF feature importance:")
    top_pos, top_neg = top_n_words(tfidf_model, tfidf_vectorizer, n=10)
    print("  Positive:", [w for w, _ in top_pos])
    print("  Negative:", [w for w, _ in top_neg])


if __name__ == "__main__":
    main()
