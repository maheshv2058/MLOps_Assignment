"""
train.py — Model loading, Trainer setup, and training loop.

Usage:
    python train.py

Environment variables:
    WANDB_API_KEY   — your Weights & Biases API key
    HF_TOKEN        — your Hugging Face token (for push_to_hub in Task 6)
"""

import os

import wandb
from transformers import (
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from data import load_and_prepare
from utils import compute_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-cased"
OUTPUT_DIR = "./results"
CACHED_MODEL_DIR = "distilbert-reviews-genres"
WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN_NAME = "distilbert-run-1"

TRAINING_CONFIG = {
    "model": MODEL_NAME,
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 3e-5,
    "max_length": 512,
    "dataset": "UCSD Goodreads",
}

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(model_name: str, num_labels: int):
    """Load a DistilBERT sequence-classification model."""
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    ).to(DEVICE)
    print(f"Model loaded: {model_name} | labels: {num_labels} | device: {DEVICE}")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    # 1. Data
    train_dataset, test_dataset, label2id, id2label, test_labels, test_texts = (
        load_and_prepare()
    )

    # 2. Model
    model = load_model(MODEL_NAME, num_labels=len(id2label))

    # 3. W&B initialisation
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=TRAINING_CONFIG,
    )

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAINING_CONFIG["epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=32,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 6. Train
    trainer.train()

    # 7. Save model locally
    trainer.save_model(CACHED_MODEL_DIR)
    print(f"Model saved to {CACHED_MODEL_DIR}")

    # 8. (Optional) Push to Hugging Face Hub — set HF_TOKEN env var first
    hf_token = os.environ.get("HF_TOKEN")
    hf_username = os.environ.get("HF_USERNAME", "your-username")
    if hf_token:
        from huggingface_hub import login
        from transformers import DistilBertTokenizerFast

        login(token=hf_token)
        repo_id = f"{hf_username}/distilbert-goodreads-genres"

        model.push_to_hub(repo_id)
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        tokenizer.push_to_hub(repo_id)
        hf_url = f"https://huggingface.co/{repo_id}"
        wandb.run.summary["huggingface_model"] = hf_url
        print(f"Model pushed to Hugging Face: {hf_url}")
    else:
        print("HF_TOKEN not set — skipping Hugging Face Hub push.")

    wandb.finish()
    return trainer, test_dataset, test_labels, id2label


if __name__ == "__main__":
    train()
