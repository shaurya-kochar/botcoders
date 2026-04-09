"""import pandas as pd

def clean_text():
    # Load raw data
    twitter = pd.read_csv("data/raw/kaggle_twitter.csv", low_memory=False)
    reddit = pd.read_csv("data/raw/kaggle_reddit.csv", low_memory=False)
    news = pd.read_csv("data/raw/kaggle_news.csv", low_memory=False)

    print("Twitter columns:", twitter.columns)
    print("Reddit columns:", reddit.columns)
    print("News columns:", news.columns)

    # -------------------------
    # TWITTER (NO TIME → DROP)
    # -------------------------
    twitter["time"] = pd.NaT

    if "text" not in twitter.columns:
        twitter["text"] = ""

    # -------------------------
    # REDDIT (GOOD DATA)
    # -------------------------
    if "title" in reddit.columns:
        reddit = reddit.rename(columns={"title": "text"})

    if "created_utc" in reddit.columns:
        reddit["time"] = pd.to_datetime(reddit["created_utc"], unit="s")

    # -------------------------
    # NEWS (FIX IMPORTANT)
    # -------------------------
    if "title" in news.columns:
        news = news.rename(columns={"title": "text"})

    if "date" in news.columns:
        news = news.rename(columns={"date": "time"})

    # -------------------------
    # ADD LIKES
    # -------------------------
    twitter["likes"] = 1
    reddit["likes"] = 1
    news["likes"] = 1

    # -------------------------
    # COMBINE
    # -------------------------
    df = pd.concat([twitter, reddit, news], ignore_index=True)

    # -------------------------
    # FIX TIME (CRITICAL)
    # -------------------------
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    # -------------------------
    # REMOVE INVALID ROWS
    # -------------------------
    df = df.dropna(subset=["time", "text"])

    print("\nSample time values:")
    print(df["time"].head())

    # -------------------------
    # SAVE CLEAN DATA
    # -------------------------
    df.to_csv("data/clean_text.csv", index=False)

    print("\n✅ Clean text ready!")

if __name__ == "__main__":
    clean_text()"""


"""
clean_text.py  —  FIXED VERSION
==================================
BUG 2 FIX: Removed the line that set twitter["time"] = pd.NaT
           which was wiping all 3M Twitter timestamps.

The 3M Twitter Kaggle file (stock_tweets.csv) has a real "Date" column:
  Format: "2021-09-01 10:00:00+00:00"  (UTC timestamp)
This script now reads it correctly.

Expected column names in kaggle_twitter.csv (stock_tweets.csv):
  Tweet       — the tweet text
  Date        — UTC timestamp
  Stock Name  — ticker symbol mentioned (AAPL, AMZN, etc.)
"""

import pandas as pd
import numpy as np

def clean_text():
    # Load raw data
    twitter = pd.read_csv(
    "data/raw/kaggle_twitter.csv",
    sep=',',
    engine='python',
    on_bad_lines='skip',
    encoding='utf-8',
    quotechar='"'
)

# clean column names
    twitter.columns = twitter.columns.str.strip()

    print("Twitter columns:", twitter.columns.tolist())

# FIX columns
    #twitter["text"] = twitter["Tweet"].astype(str)
   # twitter["date"] = pd.to_datetime(twitter["Date"], errors='coerce').dt.date
    #twitter["source"] = "twitter"
    
    reddit  = pd.read_csv("data/raw/kaggle_reddit.csv",  low_memory=False)
    news    = pd.read_csv("data/raw/kaggle_news.csv",     low_memory=False)

    print("Twitter columns:", list(twitter.columns))
    print("Reddit  columns:", list(reddit.columns))
    print("News    columns:", list(news.columns))

    # ── TWITTER ──────────────────────────────────────────────────────────
    # The 3M Kaggle dataset (stock_tweets.csv) real columns:
    #   Tweet, Date, Stock Name
    # Map to standard names
    tw_col_map = {}
    for c in twitter.columns:
        cl = c.lower().strip()
        if cl in ["tweet", "text", "content", "body"]:          tw_col_map[c] = "text"
        elif cl in ["date", "timestamp", "created_at", "time"]: tw_col_map[c] = "time"
        elif cl in ["stock name", "stock", "ticker", "symbol"]: tw_col_map[c] = "stock"

    twitter = twitter.rename(columns=tw_col_map)

    if "text" not in twitter.columns:
        print("  WARNING: No text column found in Twitter file.")
        print(f"  Available columns: {list(twitter.columns)}")
        twitter["text"] = ""

    if "time" not in twitter.columns:
        print("  WARNING: No time column found in Twitter file — timestamps will be NaT")
        twitter["time"] = pd.NaT
    else:
        # BUG 2 FIX: parse the real UTC timestamps
        twitter["time"] = pd.to_datetime(twitter["time"], errors="coerce", utc=True)
        valid_tw = twitter["time"].notna().sum()
        print(f"  Twitter: {valid_tw:,} rows have valid timestamps")

    twitter["source"] = "twitter"
    twitter["likes"]  = 1

    # ── REDDIT ───────────────────────────────────────────────────────────
    if "title" in reddit.columns:
        reddit = reddit.rename(columns={"title": "text"})
    if "created_utc" in reddit.columns:
        reddit["time"] = pd.to_datetime(reddit["created_utc"], unit="s", errors="coerce", utc=True)

    reddit["source"] = "reddit"
    reddit["likes"]  = pd.to_numeric(reddit.get("score", 1), errors="coerce").fillna(1).clip(lower=1)

    # ── NEWS ─────────────────────────────────────────────────────────────
    if "title" in news.columns:
        news = news.rename(columns={"title": "text"})
    if "date" in news.columns:
        news = news.rename(columns={"date": "time"})
        news["time"] = pd.to_datetime(news["time"], errors="coerce", utc=True)

    news["source"] = "news"
    news["likes"]  = 50

    # ── COMBINE ───────────────────────────────────────────────────────────
    df = pd.concat([twitter, reddit, news], ignore_index=True)

    # Parse time and normalise to UTC
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    # Remove rows without a timestamp or text
    df = df.dropna(subset=["time", "text"])
    df = df[df["text"].astype(str).str.strip() != ""]

    # Keep only what we need
    keep = ["text", "time", "likes", "source", "stock"]
    keep = [c for c in keep if c in df.columns]
    df   = df[keep]

    df.to_csv("data/clean_text.csv", index=False)

    print(f"\nClean text ready!")
    print(f"  Total rows  : {len(df):,}")
    print(f"  Source breakdown:")
    if "source" in df.columns:
        print(df["source"].value_counts().to_string())
    print(f"\n  Date range: {df['time'].min()} → {df['time'].max()}")

if __name__ == "__main__":
    clean_text()