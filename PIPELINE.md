# Full Pipeline — Data Processing to Score Prediction

This document describes the complete end-to-end pipeline of the **botcoders** project, from raw data ingestion to final close-price prediction.

---

## Table of Contents

- [Overview](#overview)
- [Stage 1 — Raw Data](#stage-1--raw-data)
- [Stage 2 — Stock Feature Engineering](#stage-2--stock-feature-engineering)
- [Stage 3 — Sentiment Scoring with FinBERT](#stage-3--sentiment-scoring-with-finbert)
- [Stage 4 — Dataset Assembly](#stage-4--dataset-assembly)
- [Stage 5 — Data Loading & Preprocessing](#stage-5--data-loading--preprocessing)
- [Stage 6 — Model Training](#stage-6--model-training)
- [Stage 7 — Evaluation & Reporting](#stage-7--evaluation--reporting)
- [End-to-End Flow Diagram](#end-to-end-flow-diagram)
- [Running the Full Pipeline](#running-the-full-pipeline)

---

## Overview

```
Raw CSV (news + OHLCV)
        │
        ▼
[Stage 2] yfinance → running_max, running_min
        │
        ▼
[Stage 3] FinBERT  → positiveness, negativeness
        │
        ▼
[Stage 4] full_dataset_with_sentiment.csv  (13 columns)
        │
        ▼
[Stage 5] StandardScaler + PyTorch DataLoader
        │
        ▼
[Stage 6] Linear Regression / Random Forest / LSTM
        │
        ▼
[Stage 7] Metrics (RMSE, MAE, R²) + Plots + report.md
```

---

## Stage 1 — Raw Data

**Input file:** `clean_dataset.csv`

| Column | Type | Description |
|---|---|---|
| `date` | date | Trading date |
| `text` | string | News article or tweet text |
| `sentiment` | string | Human-labelled sentiment: `positive`, `negative`, `neutral` |
| `stock_symbol` | string | Ticker: `AAPL`, `AMZN`, `GOOGL`, `MSFT`, `TSLA` |
| `open` | float | Opening price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Closing price **(prediction target)** |
| `volume` | int | Trading volume |

**Dataset size:** ~74,709 rows | Date range: 2018-01-02 → 2024-05-21

---

## Stage 2 — Stock Feature Engineering

**Script:** `datasets/build_dataset.py`

Two derived features are added by fetching historical OHLCV data from **Yahoo Finance** via `yfinance`:

| Feature | Derivation | Meaning |
|---|---|---|
| `running_max` | `cummax` of `High` over the lookback window | Highest price the stock has reached up to this date |
| `running_min` | `cummin` of `Low` over the lookback window | Lowest price the stock has reached up to this date |

**Strategy:** One bulk `yfinance.download()` per ticker (5 downloads total) covering `earliest_date − 365 days` to `latest_date`. Running stats are then merged back onto each CSV row by date.

Weekend/holiday dates in the CSV are matched to the last available trading day via forward-fill.

```
clean_dataset.csv  +  yfinance(AAPL, AMZN, GOOGL, MSFT, TSLA)
        │
        ▼
full_dataset.csv   (+2 columns: running_max, running_min)
```

---

## Stage 3 — Sentiment Scoring with FinBERT

**Script:** `datasets/add_sentiment_scores.py`

### What is FinBERT?

**FinBERT** ([ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)) is a BERT-based language model pre-trained on large financial corpora and then fine-tuned on the **Financial PhraseBank** dataset (~10,000 manually labelled financial news sentences) for sentiment classification.

| Property | Detail |
|---|---|
| Base model | `bert-base-uncased` |
| Fine-tuned on | Financial PhraseBank (Malo et al., 2014) |
| Output classes | `positive`, `negative`, `neutral` |
| Output format | Softmax probabilities (sum to 1.0 per text) |
| Max input length | 512 tokens (BERT hard limit) |
| HuggingFace ID | `ProsusAI/finbert` |

FinBERT understands **financial language** — phrases like *"earnings beat expectations"* or *"margin call"* are interpreted correctly, unlike general-purpose sentiment models.

### Scoring Process

For each row's `text`:

```
text (news article / tweet)
        │
        ▼
BertTokenizer
  • WordPiece tokenization
  • Truncate to 512 tokens
  • Pad to batch max length
  • Output: input_ids, attention_mask, token_type_ids
        │
        ▼
FinBERT (BertForSequenceClassification)
  • 12 transformer layers
  • 768 hidden dimensions
  • Classification head: Linear(768 → 3)
        │
        ▼
Softmax → [p_positive, p_negative, p_neutral]
        │
        ├──► positiveness = p_positive  (0–1)
        └──► negativeness = p_negative  (0–1)
```

### Why not use the existing `sentiment` label?

The existing `sentiment` column is a **hard label** (one of three strings). FinBERT gives **continuous probability scores**, which carry richer signal — e.g. `positiveness=0.91` vs `positiveness=0.51` both map to the label `positive` but carry very different information for the prediction model.

### Output

```
full_dataset.csv  +  FinBERT(text)
        │
        ▼
full_dataset_with_sentiment.csv  (+2 columns: positiveness, negativeness)
```

**Runtime:** ~3–5 min on GPU / ~30–60 min on CPU for 74k rows at `batch_size=32`.

---

## Stage 4 — Dataset Assembly

After Stages 2 and 3, the final dataset has **13 columns**:

| # | Column | Source |
|---|---|---|
| 1 | `date` | Raw data |
| 2 | `text` | Raw data |
| 3 | `sentiment` | Raw data (hard label) |
| 4 | `stock_symbol` | Raw data |
| 5 | `open` | Raw data |
| 6 | `high` | Raw data |
| 7 | `low` | Raw data |
| 8 | `close` | Raw data **(target)** |
| 9 | `volume` | Raw data |
| 10 | `running_max` | Stage 2 — yfinance |
| 11 | `running_min` | Stage 2 — yfinance |
| 12 | `positiveness` | Stage 3 — FinBERT |
| 13 | `negativeness` | Stage 3 — FinBERT |

---

## Stage 5 — Data Loading & Preprocessing

**Script:** `datasets/dataloader.py`

```
full_dataset_with_sentiment.csv
        │
        ▼
Sort by date (chronological — no data leakage)
        │
        ▼
Select feature columns:
  Without sentiment → [open, high, low, volume, running_max, running_min]        (6)
  With sentiment    → [open, high, low, volume, running_max, running_min,
                        positiveness, negativeness]                               (8)
        │
        ▼
train_test_split (shuffle=False, test_size=0.2)
        │
        ▼
StandardScaler  →  fit on train, transform both train & test
        │
        ▼
StockDataset (torch.utils.data.Dataset)
        │
        ▼
DataLoader  →  batch_size=4, shuffle=True (train) / False (test)
```

`shuffle=False` during splitting is critical — it preserves chronological order so the model never sees future data during training.

---

## Stage 6 — Model Training

Three models are trained in both the **without-sentiment** and **with-sentiment** configurations, giving 6 total experiments:

### Linear Regression (`models/linear_regression.py`)

- `sklearn.linear_model.LinearRegression`
- Fits a linear combination of the input features
- Baseline model — fast, interpretable, no temporal awareness

### Random Forest (`models/random_forest.py`)

- `sklearn.ensemble.RandomForestRegressor` (100 trees, full depth)
- Captures non-linear feature interactions
- Outputs feature importances — useful for understanding which features matter
- No temporal awareness (rows treated independently)

### LSTM (`models/lstm.py`)

```
Input (batch, n_features)
   → unsqueeze(1) → (batch, 1, n_features)   # seq_len = 1
   → LSTM(hidden=64, layers=2, dropout=0.2)
   → last hidden state (batch, 64)
   → Linear(64 → 1)
   → Predicted Close Price
```

- Tracks **train loss** and **validation loss** per epoch
- Designed for temporal data; extend to multi-step sequences for stronger temporal modelling

---

## Stage 7 — Evaluation & Reporting

**Script:** `evaluate_all.py` → calls `utils/visualize.py` + `utils/reporter.py`

### Metrics computed (train & test for every model × sentiment variant)

| Metric | Formula | Goal |
|---|---|---|
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i-\hat{y}_i)^2}$ | Minimise |
| MAE  | $\frac{1}{n}\sum\|y_i-\hat{y}_i\|$ | Minimise |
| R²   | $1 - \frac{SS_{res}}{SS_{tot}}$ | Maximise (≤ 1.0) |

### Plots generated (`plots/`)

| File | Content |
|---|---|
| `metrics_comparison_train.png` | Grouped bars — RMSE/MAE/R² on train set, all models |
| `metrics_comparison_test.png` | Same on test set |
| `actual_vs_predicted.png` | Actual vs predicted line plots per model × sentiment |
| `lstm_loss_curves.png` | LSTM train vs validation loss per epoch |
| `sentiment_impact.png` | Δ metric bars showing effect of adding sentiment features |

### Report

`utils/reporter.py` writes/overwrites `report.md` with metric tables, sentiment impact summary, and all embedded plot images. Re-run `evaluate_all.py` at any time to refresh it.

---

## End-to-End Flow Diagram

```
clean_dataset.csv
(date, text, sentiment, stock_symbol, open, high, low, close, volume)
        │
        ▼
datasets/build_dataset.py
  └─ yfinance bulk download (1 per ticker)
  └─ cummax(High) → running_max
  └─ cummin(Low)  → running_min
        │
        ▼
full_dataset.csv  [+running_max, +running_min]
        │
        ▼
datasets/add_sentiment_scores.py
  └─ BertTokenizer (truncate/pad to 512 tokens)
  └─ FinBERT (ProsusAI/finbert)
  └─ Softmax → [p_pos, p_neg, p_neu]
  └─ positiveness = p_pos
  └─ negativeness = p_neg
        │
        ▼
full_dataset_with_sentiment.csv  [+positiveness, +negativeness]
        │
        ▼
datasets/dataloader.py
  └─ Chronological sort → train/test split (80/20, no shuffle)
  └─ StandardScaler (fit on train only)
  └─ PyTorch DataLoader (batch_size=4)
        │
        ├──────────────────────────────────────────────┐
        ▼                   ▼                          ▼
Linear Regression     Random Forest                  LSTM
(sklearn)             (sklearn, 100 trees)    (PyTorch, 2-layer, hidden=64)
        │                   │                          │
        └───────────────────┴──────────────────────────┘
                            │
                            ▼
                  RMSE / MAE / R²  (train + test)
                            │
                  utils/visualize.py → plots/*.png
                            │
                  utils/reporter.py  → report.md
```

---

## Running the Full Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build running stats from Yahoo Finance
python datasets/build_dataset.py \
  --csv  /path/to/clean_dataset.csv \
  --out  /path/to/full_dataset.csv

# 3. Score sentiment with FinBERT
python datasets/add_sentiment_scores.py \
  --csv        /path/to/full_dataset.csv \
  --out        /path/to/full_dataset_with_sentiment.csv \
  --batch_size 32

# 4. Train all models + generate plots + update report
python evaluate_all.py \
  --csv    /path/to/full_dataset_with_sentiment.csv \
  --report report.md \
  --plots  plots/

# 5. Open report
# report.md contains all metric tables and embedded plots
```
