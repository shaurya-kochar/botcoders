"""
Centralised configuration for the EDA & Feature Engineering pipeline.
Every path, constant and symbol lives here so the rest of the codebase
stays free of magic strings.
"""

import os
from pathlib import Path

# ── Directories ────────────────────────────────────────────────────────────────

BASE_DIR   = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = BASE_DIR / "data"
PLOTS_DIR  = BASE_DIR / "plots"
OUTPUT_DIR = BASE_DIR / "outputs"

# make sure they exist on import
for _d in [DATA_DIR, PLOTS_DIR, OUTPUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Data files ─────────────────────────────────────────────────────────────────

RAW_SENTIMENT_CSV = DATA_DIR / "raw_sentiment.csv"
RAW_PRICE_CSV     = DATA_DIR / "raw_price.csv"
PROCESSED_CSV     = OUTPUT_DIR / "processed_dataset.csv"
FEATURES_CSV      = OUTPUT_DIR / "features.csv"
EDA_REPORT_TXT    = OUTPUT_DIR / "eda_report.txt"

# ── Asset seeds (same values used in the Next.js mock layer) ──────────────────

SEEDS = {
    # Stocks
    "AAPL":    {"score":  0.35, "base":    175, "volatility": 0.020},
    "TSLA":    {"score": -0.10, "base":    245, "volatility": 0.035},
    "NVDA":    {"score":  0.55, "base":    820, "volatility": 0.030},
    "AMZN":    {"score":  0.20, "base":    182, "volatility": 0.022},
    "MSFT":    {"score":  0.40, "base":    415, "volatility": 0.018},
    # Indices
    "NIFTY50": {"score":  0.25, "base": 22400, "volatility": 0.012},
    "SENSEX":  {"score":  0.22, "base": 73800, "volatility": 0.012},
    "SP500":   {"score":  0.30, "base":  4950, "volatility": 0.013},
    "NASDAQ":  {"score":  0.28, "base": 15400, "volatility": 0.016},
}

DEFAULT_SEED = {"score": 0.1, "base": 100, "volatility": 0.02}

# symbols we'll actually process in this pipeline
STOCK_SYMBOLS = ["AAPL", "TSLA", "NVDA", "AMZN", "MSFT"]
INDEX_SYMBOLS = ["NIFTY50", "SENSEX", "SP500", "NASDAQ"]
ALL_SYMBOLS   = STOCK_SYMBOLS + INDEX_SYMBOLS

# ── Generation parameters ─────────────────────────────────────────────────────

START_DATE = "2025-01-01"
NUM_DAYS   = 60
SOURCES    = ["twitter", "reddit", "instagram", "newsletter"]

# example social-media-style text fragments for each sentiment band
SAMPLE_TEXTS = {
    "very_positive": [
        "🚀🚀 This stock is going to the moon! Best buy ever!",
        "Absolutely bullish on this one, fundamentals are rock solid 💪",
        "Just loaded up more shares, earnings are gonna be insane!",
        "Best performing stock in my portfolio, no doubt about it 📈",
        "Incredible growth story. Long and strong! 💰",
    ],
    "positive": [
        "Looking good today, solid green across the board.",
        "Decent earnings beat, price should climb steadily.",
        "Market feeling optimistic about this sector overall.",
        "Newsletter says accumulate on dips, I agree.",
        "Saw some good analysis on Reddit, mildly bullish.",
    ],
    "neutral": [
        "Not sure what to make of today's price action tbh.",
        "Holding my position, nothing exciting happening rn.",
        "Market is kinda flat, waiting for catalyst.",
        "No strong opinion either way, just watching for now.",
        "Trading sideways, might break out or break down 🤷",
    ],
    "negative": [
        "This one's been bleeding for weeks, not great.",
        "Earnings miss reported, could see more downside.",
        "Sold half my position, getting nervous about macro.",
        "Newsletter flagged some concerns, worth being cautious.",
        "Sentiment on Reddit is turning sour for this ticker.",
    ],
    "very_negative": [
        "Absolute disaster, this stock is a dumpster fire 🔥🗑️",
        "Lost so much on this, worst trade of my life smh",
        "Management is clueless, selling everything ASAP 😡",
        "Stay away, this is a value trap, fundamentals are trash",
        "Bear market is here, everything is crashing hard 📉📉",
    ],
}

# ── Plotting style ─────────────────────────────────────────────────────────────

PLOT_STYLE   = "seaborn-v0_8-whitegrid"
PLOT_DPI     = 150
FIG_SIZE     = (12, 6)
PALETTE      = "coolwarm"
