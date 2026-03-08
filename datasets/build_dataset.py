"""
build_dataset.py — Load clean_dataset.csv, append real running_max and running_min
for each row using a single bulk yfinance download per ticker.

Output columns:
    date, text, sentiment, stock_symbol,
    open, high, low, close, volume,
    running_max, running_min

Usage:
    python datasets/build_dataset.py \
        --csv  /path/to/clean_dataset.csv \
        --out  /path/to/full_dataset.csv \
        --lookback 365
"""

import argparse
import os
import sys

import pandas as pd
import yfinance as yf
from datetime import timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, ".."))

OUTPUT_COLS = [
    "date", "text", "sentiment", "stock_symbol",
    "open", "high", "low", "close", "volume",
    "running_max", "running_min",
]


def _fetch_running_stats(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV for one ticker and return a DataFrame with running_max / running_min.
    Index is the trading date (normalized, timezone-naive).
    """
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No data returned for '{ticker}' ({start} → {end}).")

    # Flatten MultiIndex columns (yfinance ≥ 0.2 wraps even single ticker in MultiIndex)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.sort_index()
    raw.index = pd.to_datetime(raw.index).normalize()

    raw["running_max"] = raw["High"].cummax()
    raw["running_min"] = raw["Low"].cummin()

    return raw[["running_max", "running_min"]]


def build_dataset(
    csv_path: str,
    out_path: str,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Enrich clean_dataset.csv with running_max and running_min per row.

    For each unique stock_symbol:
      - Download OHLCV from (earliest_date - lookback_days) to latest_date
      - Compute cumulative running max (on High) and running min (on Low)
      - Forward-fill to cover weekend/holiday gaps
      - Merge back on (date, stock_symbol)

    Parameters
    ----------
    csv_path     : Input CSV path (must have date, stock_symbol columns)
    out_path     : Output enriched CSV path
    lookback_days: Calendar days before first date to start the yfinance window
    """
    print(f"\nLoading  : {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["stock_symbol", "date"]).reset_index(drop=True)

    tickers    = df["stock_symbol"].unique().tolist()
    earliest   = df["date"].min()
    latest     = df["date"].max()
    fetch_start = (earliest - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    fetch_end   = (latest   + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Tickers  : {tickers}")
    print(f"Date range: {earliest.date()}  →  {latest.date()}")
    print(f"Fetching  : {fetch_start}  →  {fetch_end}  (lookback={lookback_days}d)\n")

    stats_frames = []

    for ticker in tickers:
        print(f"  [{ticker}] downloading ...", end=" ", flush=True)
        stats = _fetch_running_stats(ticker, fetch_start, fetch_end)

        # Build a continuous daily date range and forward-fill to cover non-trading days
        full_range = pd.date_range(start=fetch_start, end=latest, freq="D", normalize=True)
        stats = stats.reindex(full_range).ffill().dropna()

        # Filter to dates actually present in the CSV for this ticker
        ticker_dates = df.loc[df["stock_symbol"] == ticker, "date"].unique()
        stats = stats[stats.index.isin(ticker_dates)]
        stats["stock_symbol"] = ticker
        stats["date"] = stats.index
        stats_frames.append(stats.reset_index(drop=True))
        print(f"{len(stats)} rows ✓")

    all_stats = pd.concat(stats_frames, ignore_index=True)

    df = df.merge(
        all_stats[["date", "stock_symbol", "running_max", "running_min"]],
        on=["date", "stock_symbol"],
        how="left",
        suffixes=("_old", ""),
    )

    # Drop old columns if they existed already in the CSV
    for col in ["running_max_old", "running_min_old"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Keep only the desired output columns (add any extra columns not in OUTPUT_COLS at end)
    extra = [c for c in df.columns if c not in OUTPUT_COLS]
    final_cols = OUTPUT_COLS + extra
    df = df[[c for c in final_cols if c in df.columns]]

    missing = df[["running_max", "running_min"]].isna().sum().sum()
    if missing:
        print(f"\nWarning: {missing} rows could not be matched to trading data (left as NaN).")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved    : {out_path}")
    print(f"Shape    : {df.shape}")
    print(f"\nSample:")
    print(df[["date", "stock_symbol", "running_max", "running_min"]].head(8).to_string(index=False))

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append running_max / running_min to clean_dataset.csv")
    parser.add_argument(
        "--csv", default="/home/akashmanna/dataset/inlp_project/clean_dataset.csv",
        help="Input CSV path"
    )
    parser.add_argument(
        "--out", default="/home/akashmanna/dataset/inlp_project/full_dataset.csv",
        help="Output CSV path (default: full_dataset.csv alongside input)"
    )
    parser.add_argument(
        "--lookback", default=365, type=int,
        help="Calendar days before first date for running stat window (default: 365)"
    )
    args = parser.parse_args()

    build_dataset(
        csv_path=args.csv,
        out_path=args.out,
        lookback_days=args.lookback,
    )
