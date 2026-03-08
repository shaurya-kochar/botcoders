"""
feature_engineering.py
─────────────────────
Reads the raw sentiment & price CSVs, computes all required features, and
writes two artefacts:

  eda/outputs/processed_dataset.csv   – merged & cleaned daily data
  eda/outputs/features.csv            – final feature matrix ready for modelling

Feature catalogue
─────────────────
  textual_polarity  Snet = (Pos - Neg) / Total   per-row polarity score
  mmi               Market Mood Index = daily mean of sentiment_score
  sentiment_vol     Sentiment volatility = std(sentiment) over the day
  sentiment_lag1    sentiment(t-1)
  sentiment_lag2    sentiment(t-2)
  price_return      daily log return of closing price
  price_vol_7d      rolling 7-day std of log return
  volume_change     day-over-day % change in volume
  text_length       character count of original text
  word_count        word count of original text
"""

import sys
import warnings

import numpy as np
import pandas as pd

from config import (
    RAW_SENTIMENT_CSV, RAW_PRICE_CSV,
    PROCESSED_CSV, FEATURES_CSV,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ─── Load ──────────────────────────────────────────────────────────────────────

def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path, label in [(RAW_SENTIMENT_CSV, "sentiment"), (RAW_PRICE_CSV, "price")]:
        if not path.exists():
            sys.exit(
                f"[ERROR] {label} CSV not found at {path}.\n"
                "       Run  python generate_data.py  first."
            )
    sent  = pd.read_csv(RAW_SENTIMENT_CSV, parse_dates=["date"])
    price = pd.read_csv(RAW_PRICE_CSV, parse_dates=["date"])
    return sent, price


# ─── Textual features (row-level) ─────────────────────────────────────────────

def add_textual_features(sent: pd.DataFrame) -> pd.DataFrame:
    """Enrich each sentiment row with text-derived features."""
    df = sent.copy()

    # text length and word count
    df["text_length"] = df["text"].str.len().fillna(0).astype(int)
    df["word_count"]  = df["text"].str.split().str.len().fillna(0).astype(int)

    # textual polarity: Snet = (Pos - Neg) / Total
    # 'Total' here means Pos + Neg + Neu (should sum to ~1.0 already)
    total = df["positive"] + df["negative"] + df["neutral"]
    # guard against division-by-zero (shouldn't happen, but be safe)
    total = total.replace(0, np.nan)
    df["textual_polarity"] = ((df["positive"] - df["negative"]) / total).round(4)
    # if total was somehow 0, fall back to the raw sentiment score
    df["textual_polarity"] = df["textual_polarity"].fillna(df["sentiment_score"])

    # human-readable sentiment class for analysis convenience
    df["sentiment_class"] = pd.cut(
        df["sentiment_score"],
        bins=[-1.01, -0.25, 0.25, 1.01],
        labels=["Negative", "Neutral", "Positive"],
    )

    return df


# ─── Daily aggregation ────────────────────────────────────────────────────────

def compute_daily_sentiment(sent: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-source rows into one row per (date, symbol) containing:
      mmi               – Market Mood Index (daily sentiment mean)
      sentiment_vol     – intraday sentiment volatility (std)
      textual_polarity  – mean Snet across sources
      post_count        – how many posts that day
    """
    agg = (
        sent.groupby(["date", "symbol"])
        .agg(
            mmi=("sentiment_score", "mean"),
            sentiment_vol=("sentiment_score", "std"),
            textual_polarity=("textual_polarity", "mean"),
            positive_mean=("positive", "mean"),
            negative_mean=("negative", "mean"),
            neutral_mean=("neutral", "mean"),
            post_count=("sentiment_score", "size"),
            avg_text_length=("text_length", "mean"),
            avg_word_count=("word_count", "mean"),
        )
        .reset_index()
    )

    # when there's only 1 post per day for a symbol, std is NaN → fill with 0
    agg["sentiment_vol"] = agg["sentiment_vol"].fillna(0).round(4)
    agg["mmi"]           = agg["mmi"].round(4)
    agg["textual_polarity"] = agg["textual_polarity"].round(4)

    return agg


# ─── Merge with price ─────────────────────────────────────────────────────────

def merge_with_price(daily_sent: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Left-join daily sentiment onto price data."""
    merged = pd.merge(price, daily_sent, on=["date", "symbol"], how="left")

    # fill any unmatched sentiment days with 0 / NaN-safe defaults
    for col in ["mmi", "sentiment_vol", "textual_polarity",
                "positive_mean", "negative_mean", "neutral_mean",
                "post_count", "avg_text_length", "avg_word_count"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    return merged


# ─── Time-series features (per symbol) ────────────────────────────────────────

def add_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each symbol group (sorted by date):
      sentiment_lag1   mmi shifted by 1 day
      sentiment_lag2   mmi shifted by 2 days
      price_return     log(close_t / close_{t-1})
      price_vol_7d     rolling 7-day std of price_return
      volume_change    (volume_t - volume_{t-1}) / volume_{t-1}
      mmi_7d_ma        7-day moving average of MMI
    """
    out_frames: list[pd.DataFrame] = []

    for symbol, grp in df.groupby("symbol"):
        g = grp.sort_values("date").copy()

        # sentiment lags
        g["sentiment_lag1"] = g["mmi"].shift(1)
        g["sentiment_lag2"] = g["mmi"].shift(2)

        # price return
        g["price_return"] = np.log(g["close"] / g["close"].shift(1))
        # handle edge cases: first row will be NaN, inf if close was 0
        g["price_return"] = g["price_return"].replace([np.inf, -np.inf], np.nan)

        # rolling price volatility (7 day)
        g["price_vol_7d"] = g["price_return"].rolling(window=7, min_periods=2).std()

        # volume day-over-day change (%)
        prev_vol = g["volume"].shift(1)
        g["volume_change"] = np.where(
            prev_vol > 0,
            (g["volume"] - prev_vol) / prev_vol,
            np.nan,
        )

        # 7-day MA of MMI
        g["mmi_7d_ma"] = g["mmi"].rolling(window=7, min_periods=1).mean()

        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)

    # round float columns for readability
    float_cols = result.select_dtypes(include="float").columns
    result[float_cols] = result[float_cols].round(4)

    return result


# ─── Cleaning & final feature matrix ──────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the columns relevant for modelling, drop rows where lag features
    are NaN (first 2 rows per symbol).
    """
    feature_cols = [
        "date", "symbol",
        # target / label
        "close", "price_return",
        # sentiment features
        "mmi", "sentiment_vol", "textual_polarity",
        "positive_mean", "negative_mean", "neutral_mean",
        # lag features
        "sentiment_lag1", "sentiment_lag2",
        # price-derived features
        "price_vol_7d", "volume_change", "mmi_7d_ma",
        # meta
        "post_count", "avg_text_length", "avg_word_count",
    ]

    existing = [c for c in feature_cols if c in df.columns]
    feat = df[existing].copy()

    # drop rows with NaN in critical lag / return columns
    lag_cols = ["sentiment_lag1", "sentiment_lag2", "price_return"]
    lag_present = [c for c in lag_cols if c in feat.columns]
    before = len(feat)
    feat = feat.dropna(subset=lag_present)
    dropped = before - len(feat)

    print(f"  Dropped {dropped} rows with NaN lag/return values "
          f"({dropped / max(before, 1) * 100:.1f}% of total)")

    return feat.reset_index(drop=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    sent, price = _load()

    print("\n── Feature Engineering ──")

    # Step 1: textual features on raw sentiment
    print("1/5  Adding textual features …")
    sent = add_textual_features(sent)

    # Step 2: daily aggregation
    print("2/5  Computing daily aggregates (MMI, sentiment vol, Snet) …")
    daily = compute_daily_sentiment(sent)

    # Step 3: merge with price
    print("3/5  Merging sentiment with price data …")
    merged = merge_with_price(daily, price)

    # Step 4: time-series features
    print("4/5  Adding time-series features (lags, returns, vol) …")
    merged = add_timeseries_features(merged)

    # save the full processed dataset
    merged.to_csv(PROCESSED_CSV, index=False)
    print(f"  → {PROCESSED_CSV}  ({len(merged):,} rows, {len(merged.columns)} cols)")

    # Step 5: build clean feature matrix
    print("5/5  Building final feature matrix …")
    features = build_feature_matrix(merged)
    features.to_csv(FEATURES_CSV, index=False)
    print(f"  → {FEATURES_CSV}  ({len(features):,} rows, {len(features.columns)} cols)")

    # quick sanity check
    print("\n── Feature Matrix Summary ──")
    print(features.describe().round(4).to_string())
    print(f"\nColumns: {list(features.columns)}")
    print(f"Null counts:\n{features.isnull().sum().to_string()}")

    print("\nFeature engineering complete ✓")


if __name__ == "__main__":
    main()
