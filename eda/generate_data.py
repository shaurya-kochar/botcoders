"""
generate_data.py
────────────────
Creates realistic synthetic datasets that mirror the structure produced by
the Next.js mockData layer, but includes social-media text snippets so that
text-length and polarity analyses make sense for the EDA.

Outputs
-------
  eda/data/raw_sentiment.csv   – one row per (date, symbol, source)
  eda/data/raw_price.csv       – daily OHLCV + sentiment overlay per symbol
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import (
    SEEDS, DEFAULT_SEED, ALL_SYMBOLS, SOURCES, SAMPLE_TEXTS,
    START_DATE, NUM_DAYS, RAW_SENTIMENT_CSV, RAW_PRICE_CSV,
)


# ─── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _date_range(start: str, days: int) -> list[str]:
    """Return a list of 'YYYY-MM-DD' strings starting from `start`."""
    base = datetime.strptime(start, "%Y-%m-%d")
    return [(base + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def _pick_text(score: float) -> str:
    """Pick a sample social-media post matching the sentiment score band."""
    if score > 0.50:
        bucket = "very_positive"
    elif score > 0.10:
        bucket = "positive"
    elif score > -0.10:
        bucket = "neutral"
    elif score > -0.50:
        bucket = "negative"
    else:
        bucket = "very_negative"
    return random.choice(SAMPLE_TEXTS[bucket])


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ─── Sentiment generation ─────────────────────────────────────────────────────

def generate_sentiment(symbols: list[str]) -> pd.DataFrame:
    """
    For each symbol and each date, produce one row per source with:
      date, symbol, source, sentiment_score, positive, negative, neutral, text
    """
    rows: list[dict] = []
    dates = _date_range(START_DATE, NUM_DAYS)

    for symbol in symbols:
        seed = SEEDS.get(symbol, DEFAULT_SEED)
        score = seed["score"]

        for day_idx, date in enumerate(dates):
            # walk the score with a small random perturbation each day
            score = _clamp(score + np.random.uniform(-0.06, 0.06), -1.0, 1.0)

            for src_idx, source in enumerate(SOURCES):
                # each source gets its own minor variation
                src_score = _clamp(
                    score + np.random.uniform(-0.08, 0.08), -1.0, 1.0
                )
                pos = _clamp((src_score + 1) / 2 * 0.7 + np.random.uniform(0, 0.1), 0.02, 0.95)
                neg = _clamp((1 - (src_score + 1) / 2) * 0.6 + np.random.uniform(0, 0.1), 0.02, 0.95)
                neu = _clamp(1 - pos - neg, 0.02, 0.95)

                rows.append({
                    "date":            date,
                    "symbol":          symbol,
                    "source":          source,
                    "sentiment_score": round(src_score, 4),
                    "positive":        round(pos, 3),
                    "negative":        round(neg, 3),
                    "neutral":         round(neu, 3),
                    "text":            _pick_text(src_score),
                })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─── Price generation ─────────────────────────────────────────────────────────

def generate_price(symbols: list[str], sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each symbol, produce daily OHLCV data aligned with the sentiment dates.
    Also attaches a daily-aggregate sentiment_overlay column.
    """
    rows: list[dict] = []
    dates = _date_range(START_DATE, NUM_DAYS)

    for symbol in symbols:
        seed = SEEDS.get(symbol, DEFAULT_SEED)
        close = float(seed["base"])
        vol   = seed["volatility"]

        is_index = symbol in {"NIFTY50", "SENSEX", "SP500", "NASDAQ"}
        vol_base  = 5_000_000_000 if is_index else 10_000_000
        vol_range = 3_000_000_000 if is_index else 40_000_000

        # daily mean sentiment for this symbol
        sym_sent = (
            sentiment_df[sentiment_df["symbol"] == symbol]
            .groupby("date")["sentiment_score"]
            .mean()
        )

        for date_str in dates:
            change = np.random.uniform(-0.48, 0.52) * close * vol
            close  = round(close + change, 2)
            open_  = round(close - np.random.uniform(-0.5, 0.5) * close * vol * 0.4, 2)
            high   = round(max(open_, close) + np.random.uniform(0, 1) * close * vol * 0.3, 2)
            low    = round(min(open_, close) - np.random.uniform(0, 1) * close * vol * 0.3, 2)
            volume = int(np.random.uniform(0, 1) * vol_range + vol_base)

            dt = pd.Timestamp(date_str)
            overlay = float(sym_sent.get(dt, 0.0))

            rows.append({
                "date":              date_str,
                "symbol":            symbol,
                "open":              open_,
                "high":              high,
                "low":               low,
                "close":             close,
                "volume":            volume,
                "sentiment_overlay": round(overlay, 4),
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating synthetic sentiment data …")
    sent_df = generate_sentiment(ALL_SYMBOLS)
    sent_df.to_csv(RAW_SENTIMENT_CSV, index=False)
    print(f"  → {RAW_SENTIMENT_CSV}  ({len(sent_df):,} rows)")

    print("Generating synthetic price data …")
    price_df = generate_price(ALL_SYMBOLS, sent_df)
    price_df.to_csv(RAW_PRICE_CSV, index=False)
    print(f"  → {RAW_PRICE_CSV}  ({len(price_df):,} rows)")

    print("Data generation complete ✓")


if __name__ == "__main__":
    main()
