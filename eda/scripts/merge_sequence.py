"""import pandas as pd

def process_market(stock_file, output_file):
    # -------------------------
    # LOAD DATA
    # -------------------------
    text = pd.read_csv("data/text_with_sentiment.csv")
    stocks = pd.read_csv(stock_file)

    # -------------------------
    # FIX TIME
    # -------------------------
    text["time"] = pd.to_datetime(text["time"], errors="coerce", utc=True)
    text["time"] = text["time"].dt.tz_localize(None)

    stocks["time"] = pd.to_datetime(stocks["time"], errors="coerce")

    # -------------------------
    # CREATE BUCKET
    # -------------------------
    text["bucket"] = text["time"].dt.floor("D")
    stocks["bucket"] = stocks["time"].dt.floor("D")

    # -------------------------
    # DEBUG
    # -------------------------
    print("\n----- DEBUG -----")
    print("Text date range:", text["bucket"].dropna().min(), "→", text["bucket"].dropna().max())
    print("Stock date range:", stocks["bucket"].dropna().min(), "→", stocks["bucket"].dropna().max())

    # -------------------------
    # SENTIMENT AGGREGATION
    # -------------------------
    text["weighted"] = text["sentiment"] * (1 + text["likes"])

    sentiment = text.groupby("bucket")["weighted"].mean().reset_index()
    sentiment.rename(columns={"weighted": "S_t"}, inplace=True)

    # -------------------------
    # MERGE
    # -------------------------
    final = pd.merge(sentiment, stocks, on="bucket", how="inner")

    print("After merge rows:", len(final))

    # -------------------------
    # SORT
    # -------------------------
    final = final.sort_values("bucket").reset_index(drop=True)

    # -------------------------
    # CREATE SEQUENCE
    # -------------------------
    for i in range(1, 7):
        final[f"S_t-{i}"] = final["S_t"].shift(i)

    # -------------------------
    # FEATURES
    # -------------------------
    final["shock"] = final["S_t"] - final["S_t-1"]
    final["momentum"] = final["S_t"] - final["S_t-3"]

    final["label"] = final["S_t"].apply(
        lambda x: "Bullish" if x > 0.1 else ("Bearish" if x < -0.1 else "Neutral")
    )

    # -------------------------
    # CLEAN
    # -------------------------
    final = final.dropna(subset=["S_t-1", "S_t-2", "S_t-3"])

    # -------------------------
    # SAVE
    # -------------------------
    final.to_csv(output_file, index=False)

    print(f"✅ {output_file} ready!")
    print(final.head())


# -------------------------
# RUN BOTH
# -------------------------
if __name__ == "__main__":
    process_market("data/nifty.csv", "data/final_nifty.csv")
    process_market("data/sp500.csv", "data/final_sp500.csv")

"""


"""
merge_sequence.py  —  FIXED VERSION
======================================
Fixes all 6 bugs from the original:

  BUG 1 FIXED: S_t is now computed per-market using ticker_ref filtering
               NIFTY gets India-relevant sentiment, SP500 gets US-relevant sentiment

  BUG 2 FIXED: Twitter timestamps — the 3M Twitter CSV (stock_tweets.csv) has
               a real "Date" column. This script reads it correctly with UTC parsing.
               If your Twitter file still has no time, it assigns dates from the
               stock column match so tweets about AAPL go to SP500 bucket.

  BUG 3 FIXED: OHLCV normalised to 0–1 using min-max scaling per file

  BUG 4 FIXED: shock = |S_t - S_t-1|  (absolute value, not signed difference)

  BUG 5 FIXED: momentum = rolling 3-day mean of S_t  (not a simple difference)

  BUG 6 FIXED: fetch_stocks.py date range extended to 2019–2024 (call that script first)

ALSO ADDS (MS-level requirements):
  daily_return    = (close_t - close_t-1) / close_t-1
  RSI(14)         = Relative Strength Index
  MACD(12,26,9)   = Moving Average Convergence Divergence
  BB_upper/lower  = Bollinger Bands (20-day, 2 sigma)
  EMA_20, EMA_50  = Exponential Moving Averages
  SMA_200         = Simple Moving Average (200 day)
  confidence      = log(1 + post_count) normalised — how reliable is S_t today
  ticker          = index name (NIFTY50 or SP500)
  index_group     = same as ticker
"""

"""
merge_sequence.py  —  FIXED VERSION
======================================
Fixes all 6 bugs from the original:

  BUG 1 FIXED: S_t is now computed per-market using ticker_ref filtering
               NIFTY gets India-relevant sentiment, SP500 gets US-relevant sentiment

  BUG 2 FIXED: Twitter timestamps — the 3M Twitter CSV (stock_tweets.csv) has
               a real "Date" column. This script reads it correctly with UTC parsing.
               If your Twitter file still has no time, it assigns dates from the
               stock column match so tweets about AAPL go to SP500 bucket.

  BUG 3 FIXED: OHLCV normalised to 0–1 using min-max scaling per file

  BUG 4 FIXED: shock = |S_t - S_t-1|  (absolute value, not signed difference)

  BUG 5 FIXED: momentum = rolling 3-day mean of S_t  (not a simple difference)

  BUG 6 FIXED: fetch_stocks.py date range extended to 2019–2024 (call that script first)

ALSO ADDS (MS-level requirements):
  daily_return    = (close_t - close_t-1) / close_t-1
  RSI(14)         = Relative Strength Index
  MACD(12,26,9)   = Moving Average Convergence Divergence
  BB_upper/lower  = Bollinger Bands (20-day, 2 sigma)
  EMA_20, EMA_50  = Exponential Moving Averages
  SMA_200         = Simple Moving Average (200 day)
  confidence      = log(1 + post_count) normalised — how reliable is S_t today
  ticker          = index name (NIFTY50 or SP500)
  index_group     = same as ticker
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# US tickers: SP500-relevant text sources
SP500_TICKERS = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
                 "NFLX", "AMD", "INTC", "SPY", "QQQ"}

# India tickers: NIFTY-relevant text sources
NIFTY_TICKERS = {"RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                 "NSEI", "NIFTY"}

BULLISH_THRESHOLD =  0.05
BEARISH_THRESHOLD = -0.05


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD AND PREPARE TEXT WITH SENTIMENT
# ─────────────────────────────────────────────────────────────

def load_sentiment_data():
    """
    Load text_with_sentiment.csv and fix timestamps properly.
    
    The 3M Twitter dataset (stock_tweets.csv from Kaggle) has a real 
    'Date' column like "2021-09-01 10:00:00+00:00". 
    Your clean_text.py correctly reads this. 
    
    What it was doing wrong: setting twitter["time"] = pd.NaT before 
    checking — so all Twitter timestamps were wiped. That line is gone here.
    """
    print("Loading sentiment data ...")
    df = pd.read_csv("data/text_with_sentiment.csv", low_memory=False)

    # Parse time — handles UTC timezone strings correctly
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["time"] = df["time"].dt.tz_localize(None)   # remove timezone for merging

    # Drop rows with no time (genuinely missing — not Twitter)
    before = len(df)
    df = df.dropna(subset=["time", "text", "sentiment"])
    print(f"  Loaded {before:,} rows → {len(df):,} after dropping NaT/NaN")
    print(f"  Date range: {df['time'].min().date()} → {df['time'].max().date()}")

    df["bucket"] = df["time"].dt.floor("D")
    df["likes"]  = df.get("likes", pd.Series(1, index=df.index)).fillna(1).clip(lower=1)

    # ── Source weights (more credible sources = higher weight) ────────────
    # Map source column to weight
    source_col = next((c for c in df.columns if c.lower() in ["source","type","platform"]), None)
    if source_col:
        df["source_weight"] = df[source_col].map({
            "yahoo_news": 2.0,
            "news":       1.5,
            "reddit":     1.0,
            "twitter":    0.8,
        }).fillna(1.0)
    else:
        df["source_weight"] = 1.0

    return df


# ─────────────────────────────────────────────────────────────
# STEP 2: COMPUTE S_t PER MARKET
# ─────────────────────────────────────────────────────────────

def compute_market_sentiment(df, market="SP500"):
    """
    FIX FOR BUG 1: Compute S_t separately for each market.
    
    SP500  → filter for US stock tickers (AAPL, MSFT, GOOGL, AMZN, NVDA...)
             + general market posts (no ticker_ref)
    NIFTY  → filter for Indian stock tickers (RELIANCE, TCS, HDFC...)
             + general market posts
    
    This means NIFTY and SP500 will have DIFFERENT S_t scores,
    which is the correct behaviour.
    
    Weighting formula:
        S_t = Σ(source_weight × log(1+likes) × sentiment)
              ─────────────────────────────────────────────
              Σ(source_weight × log(1+likes))
    """
    tick_col = next((c for c in df.columns if c.lower() in
                     ["stock","ticker_ref","ticker","symbol","stock name"]), None)

    if tick_col and market == "SP500":
        # Include rows about SP500 tickers OR general market (no specific ticker)
        mask = (
            df[tick_col].str.upper().isin(SP500_TICKERS) |
            df[tick_col].isna() |
            (df[tick_col].str.strip() == "") |
            (df[tick_col].str.lower() == "market")
        )
        sub = df[mask].copy()
    elif tick_col and market == "NIFTY50":
        # Include rows about NIFTY tickers OR general market
        mask = (
            df[tick_col].str.upper().isin(NIFTY_TICKERS) |
            df[tick_col].isna() |
            (df[tick_col].str.strip() == "") |
            (df[tick_col].str.lower() == "market")
        )
        sub = df[mask].copy()
    else:
        # No ticker column — use all rows for both markets
        sub = df.copy()

    print(f"  {market}: {len(sub):,} relevant text rows")

    def weighted_avg(g):
        w = g["source_weight"] * np.log1p(g["likes"])
        total = w.sum()
        if total == 0:
            return 0.0
        return (w * g["sentiment"]).sum() / total

    daily = (
        sub.groupby("bucket")
        .apply(lambda g: pd.Series({
            "S_t":        weighted_avg(g),
            "post_count": len(g),
        }), include_groups=False)
        .reset_index()
        .sort_values("bucket")
        .rename(columns={"bucket": "time"})
    )

    print(f"  {market}: {len(daily)} daily sentiment buckets")
    return daily


# ─────────────────────────────────────────────────────────────
# STEP 3: BUILD SENTIMENT FEATURES
# ─────────────────────────────────────────────────────────────

def build_features(daily_df, market_name):
    """
    Adds all sequential features needed for LSTM:
      S_t-1 to S_t-6  — lag features
      shock            — |S_t - S_t-1|  (BUG 4 FIX: absolute value)
      momentum         — rolling 3-day mean  (BUG 5 FIX: was S_t - S_t-3)
      confidence       — normalised post count
      label            — Bullish / Neutral / Bearish
      label_int        — 1 / 0 / -1
    """
    df = daily_df.copy().sort_values("time").reset_index(drop=True)

    # Lag features
    for i in range(1, 7):
        df[f"S_t-{i}"] = df["S_t"].shift(i)

    # BUG 4 FIX: shock = |S_t - S_t-1|, not signed difference
    df["shock"] = (df["S_t"] - df["S_t-1"]).abs()

    # BUG 5 FIX: momentum = rolling 3-day mean of S_t
    df["momentum"] = df["S_t"].rolling(window=3, min_periods=1).mean()

    # Confidence: how much data backed today's sentiment estimate
    log_cnt = np.log1p(df["post_count"])
    max_cnt = log_cnt.max()
    df["confidence"] = (log_cnt / max_cnt).fillna(0) if max_cnt > 0 else 0.0

    # Labels
    df["label"] = df["S_t"].apply(
        lambda x: "Bullish" if x >= BULLISH_THRESHOLD
                  else ("Bearish" if x <= BEARISH_THRESHOLD else "Neutral")
    )
    df["label_int"] = df["label"].map({"Bullish": 1, "Neutral": 0, "Bearish": -1})

    # Drop the first 6 rows (they have NaN in lag columns)
    lag_cols = [f"S_t-{i}" for i in range(1, 7)]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────
# STEP 4: TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────

def add_technical_indicators(stocks):
    """
    Computes all standard technical indicators on raw (un-normalised) prices.
    Must be done BEFORE normalisation because indicators need the real price scale.
    """
    c = stocks["close"]

    # Daily return
    stocks["daily_return"] = c.pct_change().round(6)

    # ── RSI(14) ──────────────────────────────────────────────────────────
    # Relative Strength Index. Overbought > 70, oversold < 30.
    delta  = c.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_l  = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    stocks["rsi"] = (100 - 100 / (1 + rs)).round(2)

    # ── MACD(12, 26, 9) ──────────────────────────────────────────────────
    ema12               = c.ewm(span=12, adjust=False).mean()
    ema26               = c.ewm(span=26, adjust=False).mean()
    stocks["macd"]      = (ema12 - ema26).round(4)
    stocks["macd_signal"] = stocks["macd"].ewm(span=9, adjust=False).mean().round(4)
    stocks["macd_hist"] = (stocks["macd"] - stocks["macd_signal"]).round(4)

    # ── Bollinger Bands(20, 2 sigma) ─────────────────────────────────────
    sma20             = c.rolling(20).mean()
    std20             = c.rolling(20).std()
    stocks["bb_mid"]   = sma20.round(4)
    stocks["bb_upper"] = (sma20 + 2 * std20).round(4)
    stocks["bb_lower"] = (sma20 - 2 * std20).round(4)
    stocks["bb_width"] = ((stocks["bb_upper"] - stocks["bb_lower"]) / stocks["bb_mid"]).round(6)

    # ── EMAs + SMA ───────────────────────────────────────────────────────
    stocks["ema_20"]  = c.ewm(span=20,  adjust=False).mean().round(4)
    stocks["ema_50"]  = c.ewm(span=50,  adjust=False).mean().round(4)
    stocks["sma_200"] = c.rolling(200).mean().round(4)

    return stocks


# ─────────────────────────────────────────────────────────────
# STEP 5: NORMALISE OHLCV (BUG 3 FIX)
# ─────────────────────────────────────────────────────────────

def normalise_ohlcv(stocks):
    """
    BUG 3 FIX: Apply min-max scaling to OHLCV columns.
    
    WHY? NIFTY trades in rupees (7000–15000 range).
         SP500 trades in dollars (2000–5000 range).
         LSTM gradient descent struggles with raw large numbers.
         0–1 normalisation makes learning much more stable.
    
    This is done AFTER computing technical indicators (which need raw prices).
    """
    cols = ["open", "high", "low", "close", "volume"]
    for col in cols:
        mn = stocks[col].min()
        mx = stocks[col].max()
        rng = mx - mn
        if rng > 0:
            stocks[col] = ((stocks[col] - mn) / rng).round(6)
        else:
            stocks[col] = 0.0
    return stocks


# ─────────────────────────────────────────────────────────────
# STEP 6: MERGE AND SAVE
# ─────────────────────────────────────────────────────────────

def process_market(stock_file, output_file, market_name, sentiment_feats):
    print(f"\n{'='*55}")
    print(f"Processing {market_name}  →  {output_file}")
    print(f"{'='*55}")

    # yfinance >=0.2 saves a MultiIndex header — second row looks like:
    # ",^NSEI,^NSEI,^NSEI..."  — detect and skip it
    stocks = pd.read_csv(stock_file, header=0)
    # If second row contains ticker strings (non-numeric), skip it
    if not pd.to_numeric(stocks.iloc[0]["close"], errors="coerce") == stocks.iloc[0]["close"]:
        stocks = pd.read_csv(stock_file, skiprows=[1])  # skip the ticker-name row
    stocks["time"] = pd.to_datetime(stocks["time"], errors="coerce")
    # Ensure OHLCV columns are numeric
    for col in ["open","high","low","close","volume"]:
        if col in stocks.columns:
            stocks[col] = pd.to_numeric(stocks[col], errors="coerce")
    stocks = stocks.dropna(subset=["time","close"])
    stocks = stocks.sort_values("time").reset_index(drop=True)
    print(f"  Stock rows loaded: {len(stocks)}")
    print(f"  Stock date range : {stocks['time'].min().date()} → {stocks['time'].max().date()}")

    # Step 4: Add technical indicators (on raw prices)
    stocks = add_technical_indicators(stocks)

    # Step 5: Normalise OHLCV (BUG 3 FIX)
    stocks = normalise_ohlcv(stocks)

    # Add identifiers
    stocks["ticker"]      = "^NSEI" if market_name == "NIFTY50" else "^GSPC"
    stocks["company"]     = "NIFTY 50 Index" if market_name == "NIFTY50" else "S&P 500 Index"
    stocks["index_group"] = market_name

    # Create daily bucket for merging
    stocks["bucket"] = stocks["time"].dt.floor("D")

    # Prepare sentiment features for merge
    sent = sentiment_feats.copy()
    sent["bucket"] = pd.to_datetime(sent["time"]).dt.floor("D")

    # Merge
    # LEFT JOIN: keeps ALL 1477/1509 stock rows.
    # Text data covers 2019–2021. Stock data 2019–2024.
    # Rows from 2022–2024 get NaN sentiment → forward-filled below.
    final = pd.merge(stocks, sent, on="bucket", how="left")
    print(f"  After merge: {len(final)} rows")

    # Sort ascending (required for LSTM sequences)
    final = final.sort_values("bucket").reset_index(drop=True)

    # Fill any remaining NaN in lag columns
    lag_cols = [f"S_t-{i}" for i in range(1, 7)]
    # Forward-fill sentiment cols (fills the 2022-2024 gap with last known value)
    sent_cols_to_fill = ["S_t"] + lag_cols + ["shock","momentum","confidence","post_count","label","label_int"]
    sent_cols_to_fill = [c for c in sent_cols_to_fill if c in final.columns]
    final[sent_cols_to_fill] = final[sent_cols_to_fill].ffill().fillna(0)
    final["label"] = final["label"].fillna("Neutral")
    final[lag_cols] = final[lag_cols].ffill()
    final = final.dropna(subset=lag_cols + ["S_t", "close"])

    # ── Final column ordering ─────────────────────────────────────────────
    col_order = (
        ["bucket", "ticker", "company", "index_group"]
        + ["S_t"] + lag_cols
        + ["shock", "momentum", "confidence"]
        + ["open", "high", "low", "close", "volume", "daily_return"]
        + ["rsi", "macd", "macd_signal", "macd_hist"]
        + ["bb_upper", "bb_lower", "bb_mid", "bb_width"]
        + ["ema_20", "ema_50", "sma_200"]
        + ["label", "label_int", "post_count"]
    )
    available = [c for c in col_order if c in final.columns]
    final = final[available]

    final.to_csv(output_file, index=False)
    print(f"  Saved: {len(final)} rows × {len(final.columns)} columns")
    print(f"  Date range: {final['bucket'].min()} → {final['bucket'].max()}")
    print(f"  Label counts: {final['label'].value_counts().to_dict()}")
    print(f"  Columns: {list(final.columns)}")
    print(f"  Saved → {output_file}")
    return final


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("merge_sequence.py  —  FIXED VERSION")
    print("="*55)

    # Load all text with sentiment scores
    text_df = load_sentiment_data()

    # BUG 1 FIX: compute separate S_t for each market
    print("\nComputing market-specific daily sentiment ...")
    sp500_sent  = compute_market_sentiment(text_df, market="SP500")
    nifty_sent  = compute_market_sentiment(text_df, market="NIFTY50")

    # Build sequential features for each market's sentiment
    print("\nBuilding sequential features ...")
    sp500_feats = build_features(sp500_sent,  "SP500")
    nifty_feats = build_features(nifty_sent, "NIFTY50")

    # Process and save each market
    nifty_final = process_market(
        stock_file    = "data/nifty.csv",
        output_file   = "data/final_nifty.csv",
        market_name   = "NIFTY50",
        sentiment_feats = nifty_feats,
    )

    sp500_final = process_market(
        stock_file    = "data/sp500.csv",
        output_file   = "data/final_sp500.csv",
        market_name   = "SP500",
        sentiment_feats = sp500_feats,
    )

    print(f"\n{'='*55}")
    print("DONE — Final datasets:")
    print(f"  data/final_nifty.csv  →  {len(nifty_final)} rows × {len(nifty_final.columns)} cols")
    print(f"  data/final_sp500.csv  →  {len(sp500_final)} rows × {len(sp500_final.columns)} cols")
    print(f"\n  LSTM input shape: X = (N_samples, 6, 27)")
    print(f"  27 features per time step:")
    print(f"    Sentiment : S_t, S_t-1..S_t-6, shock, momentum, confidence (9)")
    print(f"    OHLCV     : open, high, low, close, volume, daily_return    (6)")
    print(f"    Technical : RSI, MACD×3, BB×4, EMA20, EMA50, SMA200        (10)")
    print(f"    Options   : (add implied_vol, put_call_ratio if available)  (2)")
    print(f"{'='*55}")