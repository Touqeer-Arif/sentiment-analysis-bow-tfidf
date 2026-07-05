"""
preprocessing.py
----------------
Purpose: Implements the complete NLP preprocessing pipeline required by
Task 3 of the assignment:
    1. Lowercasing
    2. Removal of punctuation
    3. Removal of special characters / digits
    4. Tokenization
    5. Stop-word removal
    6. Lemmatization

Each step is exposed as its own function so that the effect of every
individual step can be inspected (see Task 3's "show output after each
step" requirement), and a single `preprocess_pipeline()` function chains
them together for use by the feature-extraction / training modules.
"""

import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()
_stopword_set = None  # lazy-loaded so utils.py can set the NLTK path first


def _get_stopwords():
    global _stopword_set
    if _stopword_set is None:
        _stopword_set = set(stopwords.words("english"))
    return _stopword_set


# ------------------------------------------------------------------
# Individual pipeline steps (each documented for the viva / report)
# ------------------------------------------------------------------

def to_lowercase(text: str) -> str:
    """Step 1: Lowercasing.
    Purpose: Ensures 'Great' and 'great' are treated as the same token,
    preventing the vocabulary from being needlessly duplicated by case."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Step 2: Remove punctuation.
    Purpose: Punctuation marks (.,!?;:'"()...) carry little standalone
    meaning for BoW/TF-IDF and would otherwise be treated as tokens."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_special_characters(text: str) -> str:
    """Step 3: Remove special characters and digits.
    Purpose: Strips residual HTML artefacts (e.g. line breaks), numbers,
    and any non-alphabetic symbols that add noise rather than sentiment
    signal."""
    text = re.sub(r"<br\s*/?>", " ", text)          # leftover HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text)         # keep letters/spaces only
    text = re.sub(r"\s+", " ", text).strip()         # collapse whitespace
    return text


def tokenize(text: str):
    """Step 4: Tokenization.
    Purpose: Splits the continuous string of text into a list of
    individual word tokens, the unit that BoW/TF-IDF operate on."""
    return word_tokenize(text)


def remove_stopwords(tokens):
    """Step 5: Stop-word removal.
    Purpose: Removes very high-frequency, low-information words
    (e.g. 'the', 'is', 'and') that appear in almost every document and
    would otherwise dilute the discriminative vocabulary."""
    sw = _get_stopwords()
    return [tok for tok in tokens if tok not in sw and len(tok) > 1]


def lemmatize(tokens):
    """Step 6: Lemmatization.
    Purpose: Reduces inflected words to their dictionary/base form
    (e.g. 'movies' -> 'movie', 'running' -> 'running' as verb base
    'run'), so that different surface forms of a word contribute to the
    same feature instead of being counted as separate vocabulary
    entries."""
    return [_lemmatizer.lemmatize(tok) for tok in tokens]


def preprocess_pipeline(text: str) -> str:
    """
    Runs the full pipeline end-to-end and returns a single cleaned
    string (tokens re-joined by spaces) ready to be fed into
    CountVectorizer / TfidfVectorizer.
    """
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def show_pipeline_steps(text: str) -> dict:
    """
    Utility used in the notebook/report for Task 3: returns an ordered
    dictionary showing the output of the review after every single
    preprocessing step, for one example review.
    """
    steps = {}
    steps["0_original"] = text
    step1 = to_lowercase(text)
    steps["1_lowercased"] = step1
    step2 = remove_punctuation(step1)
    steps["2_punctuation_removed"] = step2
    step3 = remove_special_characters(step2)
    steps["3_special_chars_removed"] = step3
    step4 = tokenize(step3)
    steps["4_tokenized"] = step4
    step5 = remove_stopwords(step4)
    steps["5_stopwords_removed"] = step5
    step6 = lemmatize(step5)
    steps["6_lemmatized"] = step6
    steps["7_final_joined"] = " ".join(step6)
    return steps
