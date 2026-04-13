"""Text cleaning and sklearn pipeline construction."""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def clean_text(text):
    """
    Lowercase, strip HTML, keep letters and spaces only, collapse whitespace.
    """
    if text is None:
        return ""
    s = str(text).lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_pipeline(classifier):
    """
    Build a Pipeline: TfidfVectorizer (with clean_text preprocessor) + classifier.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        stop_words="english",
    )
    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", classifier),
        ]
    )
