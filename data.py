"""
data.py — Data loading, sampling, train/test split, and encoding.

Usage:
    python data.py
"""

import gzip
import json
import pickle
import random
import requests

from transformers import DistilBertTokenizerFast

from utils import MyDataset, build_label_maps

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
SAMPLE_SIZE = 1000       # reviews per genre kept after streaming
TRAIN_PER_GENRE = 800   # first N go to train, rest to test
PICKLE_PATH = "genre_reviews_dict.pickle"

GENRE_URL_DICT = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


# ---------------------------------------------------------------------------
# Streaming loader
# ---------------------------------------------------------------------------
def load_reviews(url: str, head: int = 10000, sample_size: int = 2000) -> list[str]:
    """Stream reviews from a gzipped JSON-lines URL and return a random sample."""
    reviews = []
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if head is not None and i >= head:
                break
            d = json.loads(line)
            reviews.append(d["review_text"])

    return random.sample(reviews, min(sample_size, len(reviews)))


def fetch_all_reviews(pickle_path: str = PICKLE_PATH) -> dict[str, list[str]]:
    """Load from pickle if it exists; otherwise download and save."""
    try:
        with open(pickle_path, "rb") as fh:
            print(f"Loaded cached reviews from {pickle_path}")
            return pickle.load(fh)
    except FileNotFoundError:
        pass

    genre_reviews_dict: dict[str, list[str]] = {}
    for genre, url in GENRE_URL_DICT.items():
        print(f"Downloading reviews for genre: {genre}")
        genre_reviews_dict[genre] = load_reviews(url, head=10000, sample_size=SAMPLE_SIZE)

    with open(pickle_path, "wb") as fh:
        pickle.dump(genre_reviews_dict, fh)
    print(f"Saved reviews to {pickle_path}")
    return genre_reviews_dict


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
def split_data(
    genre_reviews_dict: dict[str, list[str]],
    train_per_genre: int = TRAIN_PER_GENRE,
    sample_size: int = SAMPLE_SIZE,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (train_texts, train_labels, test_texts, test_labels)."""
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews_dict.items():
        reviews = random.sample(reviews, min(sample_size, len(reviews)))
        for review in reviews[:train_per_genre]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in reviews[train_per_genre:]:
            test_texts.append(review)
            test_labels.append(genre)

    print(
        f"Train: {len(train_texts)} reviews | Test: {len(test_texts)} reviews"
    )
    return train_texts, train_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode_data(
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
) -> tuple[MyDataset, MyDataset, dict, dict]:
    """Tokenize texts and encode labels; return dataset objects and label maps."""
    from utils import build_label_maps

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    label2id, id2label = build_label_maps(train_labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    train_labels_enc = [label2id[y] for y in train_labels]
    test_labels_enc = [label2id[y] for y in test_labels]

    train_dataset = MyDataset(train_encodings, train_labels_enc)
    test_dataset = MyDataset(test_encodings, test_labels_enc)

    return train_dataset, test_dataset, label2id, id2label


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def load_and_prepare(
    pickle_path: str = PICKLE_PATH,
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
    train_per_genre: int = TRAIN_PER_GENRE,
    sample_size: int = SAMPLE_SIZE,
) -> tuple[MyDataset, MyDataset, dict, dict, list[str], list[str]]:
    """Full pipeline: download -> split -> encode. Returns datasets and label maps."""
    genre_reviews_dict = fetch_all_reviews(pickle_path)
    train_texts, train_labels, test_texts, test_labels = split_data(
        genre_reviews_dict, train_per_genre=train_per_genre, sample_size=sample_size
    )
    train_dataset, test_dataset, label2id, id2label = encode_data(
        train_texts, train_labels, test_texts, test_labels,
        model_name=model_name, max_length=max_length,
    )
    return train_dataset, test_dataset, label2id, id2label, test_labels, test_texts


if __name__ == "__main__":
    load_and_prepare()
    print("Data pipeline complete.")
