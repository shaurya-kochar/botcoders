"""
add_sentiment_scores.py — Use FinBERT to compute positiveness and negativeness
scores for each row's text and append them to the dataset CSV.

FinBERT (ProsusAI/finbert) outputs three probabilities per text:
    positive, negative, neutral

We append:
    positiveness  — probability of positive sentiment
    negativeness  — probability of negative sentiment

Usage:
    python datasets/add_sentiment_scores.py \
        --csv  /home/akashmanna/dataset/inlp_project/full_dataset.csv \
        --out  /home/akashmanna/dataset/inlp_project/full_dataset_with_sentiment.csv \
        --batch_size 32
"""

import argparse
import os
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, ".."))

FINBERT_MODEL = "ProsusAI/finbert"
# FinBERT label order as defined in its config: positive=0, negative=1, neutral=2
LABEL_ORDER   = ["positive", "negative", "neutral"]
MAX_TOKENS    = 512     # BERT hard limit


def load_finbert(device: torch.device):
    print(f"  Loading FinBERT ({FINBERT_MODEL}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()
    model.to(device)
    print(f"  Model loaded on {device}")
    return tokenizer, model


def score_batch(texts: list[str], tokenizer, model, device: torch.device) -> tuple[list[float], list[float]]:
    """
    Run FinBERT on a batch of texts.
    Returns (positiveness_list, negativeness_list).
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits          # (batch, 3)

    probs = softmax(logits, dim=-1).cpu()         # (batch, 3)

    # Determine label → index mapping from model config
    id2label = model.config.id2label             # {0: 'positive', 1: 'negative', 2: 'neutral'}
    label2id = {v.lower(): k for k, v in id2label.items()}

    pos_idx = label2id.get("positive", 0)
    neg_idx = label2id.get("negative", 1)

    positiveness = probs[:, pos_idx].tolist()
    negativeness = probs[:, neg_idx].tolist()
    return positiveness, negativeness


def add_sentiment_scores(
    csv_path: str,
    out_path: str,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Load CSV, run FinBERT on every text, append positiveness & negativeness columns.

    Parameters
    ----------
    csv_path   : Input CSV (must have a 'text' column)
    out_path   : Where to save the enriched CSV
    batch_size : Number of texts per FinBERT forward pass (reduce if OOM)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"Loading CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    texts = df["text"].fillna("").astype(str).tolist()
    n = len(texts)
    print(f"Rows       : {n}")
    print(f"Batch size : {batch_size}\n")

    tokenizer, model = load_finbert(device)

    all_pos, all_neg = [], []

    for i in tqdm(range(0, n, batch_size), desc="FinBERT scoring"):
        batch_texts = texts[i : i + batch_size]
        pos, neg = score_batch(batch_texts, tokenizer, model, device)
        all_pos.extend(pos)
        all_neg.extend(neg)

    df["positiveness"] = [round(x, 6) for x in all_pos]
    df["negativeness"] = [round(x, 6) for x in all_neg]

    # Ensure column order: existing cols + new sentiment scores at the end
    base_cols = [c for c in df.columns if c not in ("positiveness", "negativeness")]
    df = df[base_cols + ["positiveness", "negativeness"]]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved : {out_path}")
    print(f"Shape : {df.shape}")
    print(f"\nSample (positiveness / negativeness):")
    print(df[["date", "stock_symbol", "sentiment", "positiveness", "negativeness"]].head(8).to_string(index=False))

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append FinBERT positiveness/negativeness scores to dataset CSV"
    )
    parser.add_argument(
        "--csv", default="/home/akashmanna/dataset/inlp_project/full_dataset.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--out", default="/home/akashmanna/dataset/inlp_project/full_dataset_with_sentiment.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--batch_size", default=32, type=int,
        help="Texts per FinBERT batch (reduce to 8 if GPU OOM, default: 32)"
    )
    args = parser.parse_args()

    add_sentiment_scores(
        csv_path=args.csv,
        out_path=args.out,
        batch_size=args.batch_size,
    )
