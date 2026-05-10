# MLOps Assignment 2 — DistilBERT Goodreads Genre Classifier

Fine-tuning DistilBERT on the UCSD Goodreads dataset to predict book genres from review text. Experiment tracking is done with Weights & Biases and the trained model is published to Hugging Face Hub.

---

## Project Description

This project demonstrates a complete MLOps workflow: a pre-trained transformer model (DistilBERT) is fine-tuned for multi-class text classification. The dataset consists of book reviews from the UCSD Goodreads corpus spanning seven genres (poetry, comics & graphic, fantasy & paranormal, history & biography, mystery/thriller/crime, romance, and young adult). All training runs are logged to Weights & Biases for reproducibility, and the final model weights are hosted publicly on Hugging Face Hub.

---

## Project Structure

```
.
├── data.py            # Data download, sampling, split, and tokenisation
├── train.py           # Model loading, Trainer setup, W&B logging, HF Hub push
├── eval.py            # Evaluation, classification report, W&B Artifact upload
├── utils.py           # Shared helpers: label maps, dataset class, compute_metrics
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Setup Instructions

**1. Clone the repository**

```bash
git clone https://github.com/maheshv2058/MLOps_Assignment.git
cd MLOps_Assignment
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set environment variables**

```bash
export WANDB_API_KEY=your_wandb_key
export HF_TOKEN=your_hf_token          # only needed for Hub push
export HF_USERNAME=your-hf-username    # only needed for Hub push
```

**4. Run the pipeline**

```bash
# Step 1 — Download and prepare data (also called automatically by train.py)
python data.py

# Step 2 — Fine-tune and log to W&B
python train.py

# Step 3 — Evaluate and upload report artifact to W&B
python eval.py
```

GPU is strongly recommended. On free-tier Google Colab (T4), training takes roughly 5-10 minutes. On CPU, reduce `SAMPLE_SIZE` in `data.py` to 200 and expect longer runtimes.

---

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.XX   |
| F1 Score  | 0.XX   |
| Eval Loss | 0.XX   |

*(Replace the placeholders above with your actual scores after running `eval.py`.)*

---

## Links

- Hugging Face model: https://huggingface.co/maheshvgv/distilbert-goodreads-genres
- W&B dashboard: https://wandb.ai/g25ait2058-mahesh/mlops-assignment2
