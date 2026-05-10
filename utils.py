"""
utils.py — Shared helpers: label maps, dataset class, compute_metrics
"""

import torch
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------------
# Label maps (populated at runtime by data.py / train.py)
# ---------------------------------------------------------------------------
label2id: dict = {}
id2label: dict = {}


def build_label_maps(labels: list[str]) -> tuple[dict, dict]:
    """Build label <-> integer maps from a list of string labels."""
    unique = sorted(set(labels))
    l2i = {label: idx for idx, label in enumerate(unique)}
    i2l = {idx: label for label, idx in l2i.items()}
    return l2i, i2l


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class MyDataset(torch.utils.data.Dataset):
    """Wraps HuggingFace tokenizer encodings and integer labels."""

    def __init__(self, encodings, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred) -> dict:
    """Return accuracy and weighted F1 for the HuggingFace Trainer."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }
