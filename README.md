# Comparative Sentiment Analysis using Bag of Words and TF-IDF

**Programming for AI — Assignment 4**

A complete NLP sentiment analysis pipeline that compares **Bag of Words
(BoW)** and **TF-IDF** feature extraction techniques for classifying
movie reviews as positive or negative, using Logistic Regression as the
classifier.

## Project Overview

This project implements, end to end:
- A full NLP preprocessing pipeline (lowercasing, punctuation/special
  character removal, tokenization, stop-word removal, lemmatization)
- Two feature extraction techniques: Bag of Words and TF-IDF
- Two Logistic Regression classifiers (one per feature technique)
- A full evaluation and comparison (Accuracy, Precision, Recall, F1,
  Confusion Matrix)
- Feature importance analysis (top sentiment-driving words)
- Custom review predictions on five hand-written reviews
- A written comparative analysis of BoW vs TF-IDF

## Dataset Description

**Source:** NLTK Movie Reviews Corpus (`nltk.corpus.movie_reviews`),
Option 1 (recommended) from the assignment brief.

- 2000 movie reviews total
- 1000 positive / 1000 negative (perfectly balanced)
- Bundled locally under `data/nltk_data/` so the notebook runs without
  needing a fresh internet download

## Project Structure

```
sentiment_analysis_project/
│
├── data/nltk_data/           # Bundled NLTK corpora (movie_reviews, stopwords, wordnet, punkt)
├── src/
│   ├── preprocessing.py      # Full NLP preprocessing pipeline
│   ├── feature_extraction.py # BoW + TF-IDF vectorizers
│   ├── train_bow.py          # Trains/saves the BoW Logistic Regression model
│   ├── train_tfidf.py        # Trains/saves the TF-IDF Logistic Regression model
│   ├── evaluation.py         # Metrics, comparison table, feature importance
│   └── utils.py              # Data loading, NLTK path setup, pickle helpers
│
├── notebooks/
│   └── sentiment_analysis.ipynb  # End-to-end notebook (Tasks 1-10)
│
├── results/
│   ├── plots/                # class_distribution.png, confusion_matrices.png,
│   │                         # metric_comparison.png, feature_importance.png
│   ├── models/                # bow_model.pkl, tfidf_model.pkl
│   └── tables/                 # comparison_table.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation Instructions

```bash
git clone https://github.com/Touqeer-Arif/sentiment-analysis-bow-tfidf.git
cd sentiment-analysis-bow-tfidf
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/sentiment_analysis.ipynb
```

The NLTK corpora required (`movie_reviews`, `stopwords`, `wordnet`,
`omw-1.4`, `punkt`) are already bundled under `data/nltk_data/`, and
`src/utils.py` automatically points NLTK at that folder, so no separate
`nltk.download()` step is required to run the notebook.

## Methodology

1. **Data loading** — load all 2000 labelled reviews from the NLTK corpus.
2. **Preprocessing** — lowercase → remove punctuation → remove special
   characters/digits → tokenize → remove stop-words → lemmatize.
3. **Train/test split** — 80/20 stratified split on sentiment label.
4. **Feature extraction** — fit `CountVectorizer` (BoW) and
   `TfidfVectorizer` (TF-IDF) on the training text only
   (`max_features=5000` for both, to keep the comparison fair).
5. **Model training** — train one `LogisticRegression(max_iter=1000)`
   model per feature set.
6. **Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion
   Matrix for both models, and a side-by-side comparison table/chart.
7. **Feature importance** — top 10 positive/negative words from the
   TF-IDF model's Logistic Regression coefficients.
8. **Custom predictions** — 5 hand-written reviews classified with
   confidence scores.

## Results Summary

| Metric    | BoW    | TF-IDF |
|-----------|--------|--------|
| Accuracy  | 0.8300 | 0.8225 |
| Precision | 0.8367 | 0.8116 |
| Recall    | 0.8200 | 0.8400 |
| F1 Score  | 0.8283 | 0.8256 |

*(Exact figures may shift slightly if the notebook is re-run, since the
train/test split — though seeded — depends on library versions.)*

On this dataset and split, BoW and TF-IDF performed almost identically,
with BoW holding a marginal edge overall. See Task 9 in the notebook for
the full discussion of why, and the advantages/limitations of each
technique.

## Screenshots

See `results/plots/`:
- `class_distribution.png` — positive vs negative review counts
- `confusion_matrices.png` — BoW vs TF-IDF confusion matrices
- `metric_comparison.png` — Accuracy/Precision/Recall/F1 bar chart
- `feature_importance.png` — top 10 positive/negative sentiment words


