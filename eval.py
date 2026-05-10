"""
eval.py — Evaluation, metrics, and saving results as a W&B Artifact.

Usage:
    python eval.py

Run this after train.py has saved the model to CACHED_MODEL_DIR.
"""

import json
import os

import wandb
from sklearn.metrics import classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data import load_and_prepare
from utils import MyDataset, compute_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CACHED_MODEL_DIR = "distilbert-reviews-genres"
EVAL_REPORT_PATH = "eval_report.json"
WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN_NAME = "distilbert-eval-1"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate():
    # 1. Re-load data (same random seed would give the same split; here we
    #    reload so eval.py can be run independently of train.py).
    train_dataset, test_dataset, label2id, id2label, test_labels, test_texts = (
        load_and_prepare()
    )

    # 2. Load saved model
    if not os.path.isdir(CACHED_MODEL_DIR):
        raise FileNotFoundError(
            f"Trained model not found at '{CACHED_MODEL_DIR}'. "
            "Run train.py first."
        )
    model = DistilBertForSequenceClassification.from_pretrained(CACHED_MODEL_DIR)
    print(f"Loaded model from {CACHED_MODEL_DIR}")

    # 3. Minimal TrainingArguments needed for Trainer.evaluate / predict
    eval_args = TrainingArguments(
        output_dir="./eval_output",
        per_device_eval_batch_size=32,
        report_to=[],  # no extra logging here
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 4. W&B run for eval metrics
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

    # 5. Evaluate
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # 6. Log final metrics explicitly
    wandb.log(
        {
            "final/loss": eval_results.get("eval_loss"),
            "final/accuracy": eval_results.get("eval_accuracy"),
            "final/f1": eval_results.get("eval_f1"),
        }
    )

    # 7. Full classification report
    predicted_output = trainer.predict(test_dataset)
    predicted_ids = predicted_output.predictions.argmax(-1).flatten().tolist()
    predicted_labels = [id2label[i] for i in predicted_ids]

    report = classification_report(
        test_labels,
        predicted_labels,
        target_names=list(id2label.values()),
        output_dict=True,
    )
    print("\nClassification Report:")
    print(
        classification_report(
            test_labels, predicted_labels, target_names=list(id2label.values())
        )
    )

    # 8. Save report to file
    with open(EVAL_REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved classification report to {EVAL_REPORT_PATH}")

    # 9. Upload as W&B Artifact
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file(EVAL_REPORT_PATH)
    wandb.log_artifact(artifact)
    print("Uploaded eval report to W&B as artifact.")

    wandb.finish()
    return eval_results, report


if __name__ == "__main__":
    evaluate()
