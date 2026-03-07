# botcoders

Stock close-price prediction using traditional ML and deep learning models, with and without Twitter sentiment features.

---

## Table of Contents

- [Task Description](#task-description)
- [Features](#features)
- [Architecture](#architecture)
- [Metrics](#metrics)
- [File Structure](#file-structure)
- [How to Run](#how-to-run)
- [Results](#results)

---

## Task Description

Predict the **closing price** of a stock within a given time span using historical price data and, optionally, aggregated sentiment scores from tweets.

Two setups are compared throughout:

| Setup | Features |
|---|---|
| **Without Sentiment** | open, high, low, volume, running\_max, running\_min |
| **With Sentiment** | above 6 + positiveness, negativeness |

---

## Features

### Stock Features (6)

| Feature | Description |
|---|---|
| `open` | Opening stock price |
| `high` | Highest price in the time span |
| `low` | Lowest price in the time span |
| `volume` | Total trading volume |
| `running_max` | Running maximum price up to that day |
| `running_min` | Running minimum price up to that day |

### Sentiment Features (2)

Tweets within each time span are collected and processed through a sentiment pipeline. The **average** positive and negative scores across all tweets are used:

| Feature | Description |
|---|---|
| `positiveness` | Mean positive sentiment score across tweets (0–1) |
| `negativeness` | Mean negative sentiment score across tweets (0–1) |

> Note: `positiveness + negativeness ≠ 1` — tweets can carry neutral sentiment as well.

**Target:** `close` — closing price

---

## Architecture

### Pipeline Overview

```
Raw Tweets
    │
    ▼
Sentiment Model  ──►  positiveness (avg)
                  ──►  negativeness (avg)
                              │
Stock Price Data              │
  open, high, low,            │
  volume,                     │
  running_max,                │
  running_min  ───────────────┤
                              ▼
                   Feature Vector (8-dim)
                              │
                     StandardScaler
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
     Linear Regression   Random Forest     LSTM
               │              │              │
               └──────────────┴──────────────┘
                              ▼
                    Predicted Close Price
```

### LSTM Architecture

```
Input (batch, 8)
   → unsqueeze(dim=1) → (batch, seq_len=1, 8)
   → LSTM (2 layers, hidden=64, dropout=0.2)
   → last hidden state (batch, 64)
   → Linear(64 → 1)
   → Predicted Close Price (batch, 1)
```

| Hyperparameter | Value |
|---|---|
| Hidden size | 64 |
| Layers | 2 |
| Dropout | 0.2 |
| Optimizer | Adam |
| Loss | MSELoss |
| Epochs | 100 |
| Batch size | 4 |

[View Interactive Architecture Diagram](assets/model_arch.html)

---

## Metrics

Three metrics are computed for every model on both train and test splits:

| Metric | Formula | Interpretation |
|---|---|---|
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Lower is better. In the same unit as the target (price). |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Lower is better. Average absolute error in price units. |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Higher is better. 1.0 = perfect, 0.0 = predicts mean, < 0 = worse than mean baseline. |

> A negative R² on small test sets is expected — it indicates the model generalises poorly on few samples, not that it is fundamentally broken.

---

## File Structure

```
botcoders/
│
├── samples/
│   └── dataset.csv             # Sample dataset with 20 rows (8 features + target)
│
├── datasets/
│   └── dataloader.py           # StockDataset + get_dataloaders() — train/test split,
│                               #   StandardScaler, PyTorch DataLoader
│
├── models/
│   ├── linear_regression.py    # Sklearn LinearRegression wrapper; returns metrics dict
│   ├── random_forest.py        # Sklearn RandomForestRegressor; returns metrics + feature importances
│   └── lstm.py                 # PyTorch LSTMRegressor; tracks train & val loss per epoch
│
├── utils/
│   ├── visualize.py            # Generates 5 matplotlib plots (PNG) saved to plots/
│   └── reporter.py             # Writes / overwrites report.md with tables + embedded plots
│
├── plots/                      # Auto-generated plot PNGs (created by evaluate_all.py)
│   ├── metrics_comparison_train.png
│   ├── metrics_comparison_test.png
│   ├── actual_vs_predicted.png
│   ├── lstm_loss_curves.png
│   └── sentiment_impact.png
│
├── evaluate_all.py             # Master script — runs all models, generates plots, writes report
├── train.py                    # (reserved for custom training runs)
├── report.md                   # Auto-generated evaluation report (do not edit manually)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run full evaluation (all models + plots + report)

```bash
python evaluate_all.py
```

Optional arguments:

```bash
python evaluate_all.py \
  --csv    samples/dataset.csv \   # path to dataset
  --report report.md \             # output report path
  --plots  plots/ \                # output directory for PNGs
  --epochs 100                     # LSTM training epochs
```

### 3. Run individual models

```bash
# Linear Regression only
python models/linear_regression.py

# Random Forest only
python models/random_forest.py

# LSTM only
python models/lstm.py
```

### 4. View the report

Open [report.md](report.md) — it contains metric tables and all plots, auto-updated every time `evaluate_all.py` is run.

---

## Results

See [report.md](report.md) for the full evaluation report including:

- Training and test RMSE / MAE / R² for all models
- Sentiment feature impact (Δ metrics)
- LSTM training vs validation loss curves
- Actual vs predicted plots per model

### Plots

| Plot | Description |
|---|---|
| [metrics_comparison_train.png](plots/metrics_comparison_train.png) | Grouped bar chart of train metrics across all models |
| [metrics_comparison_test.png](plots/metrics_comparison_test.png) | Grouped bar chart of test metrics across all models |
| [actual_vs_predicted.png](plots/actual_vs_predicted.png) | Actual vs predicted close price on test set |
| [lstm_loss_curves.png](plots/lstm_loss_curves.png) | LSTM train vs validation loss per epoch |
| [sentiment_impact.png](plots/sentiment_impact.png) | Δ metrics showing effect of adding sentiment features |
