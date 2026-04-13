"""Load and clean Amazon Reviews data from HuggingFace."""

import pandas as pd
from datasets import load_dataset


def load_and_clean_data():
    """
    Load All_Beauty reviews from HuggingFace, clean, sample, and add binary labels.

    Returns
    -------
    pd.DataFrame
        Columns: text, rating, label (0=negative, 1=positive).
    """
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_All_Beauty",
        split="full",
        trust_remote_code=True,
    )

    df = dataset.to_pandas()
    df = df[["text", "rating"]].copy()
    df = df.dropna(subset=["rating"])

    df = df[df["text"].notna()]
    df = df[df["text"].astype(str).str.strip() != ""]

    if len(df) > 50_000:
        df = df.sample(n=50_000, random_state=42)

    df["label"] = df["rating"].apply(lambda r: 0 if r <= 3 else 1)

    print("Class distribution after mapping:")
    print(df["label"].value_counts().sort_index())

    return df.reset_index(drop=True)


def load_data():
    """Binary sentiment loader (alias for :func:`load_and_clean_data`)."""
    return load_and_clean_data()


def load_data_ternary():
    """
    Same loading as binary, with 3-class labels from star ratings.

    Returns
    -------
    pd.DataFrame
        Columns: text, rating, label (0=Negative, 1=Neutral, 2=Positive).
    """
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_All_Beauty",
        split="full",
        trust_remote_code=True,
    )

    df = dataset.to_pandas()
    df = df[["text", "rating"]].copy()
    df = df.dropna(subset=["rating"])

    df = df[df["text"].notna()]
    df = df[df["text"].astype(str).str.strip() != ""]

    if len(df) > 50_000:
        df = df.sample(n=50_000, random_state=42)

    def rating_to_label(r):
        if r <= 2:
            return 0
        if r == 3:
            return 1
        return 2

    df["label"] = df["rating"].apply(rating_to_label)

    print("Class distribution after mapping:")
    print(df["label"].value_counts().sort_index())

    return df.reset_index(drop=True)
