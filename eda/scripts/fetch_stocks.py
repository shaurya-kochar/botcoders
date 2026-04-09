"""import yfinance as yf
import pandas as pd

def fetch_stocks():
    start = "2019-01-01"
    end = "2021-02-20"

    # NIFTY
    nifty = yf.download("^NSEI", start=start, end=end)
    nifty.reset_index(inplace=True)

    nifty.rename(columns={
        "Date": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)

    nifty.to_csv("data/nifty.csv", index=False)
    print("✅ NIFTY saved!")

    # S&P 500
    sp500 = yf.download("^GSPC", start=start, end=end)
    sp500.reset_index(inplace=True)

    sp500.rename(columns={
        "Date": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)

    sp500.to_csv("data/sp500.csv", index=False)
    print("✅ SP500 saved!")

if __name__ == "__main__":
    fetch_stocks()
    """

"""
fetch_stocks.py  —  FIXED VERSION
====================================
BUG 6 FIX: Extended date range from 2019-01-01 to 2024-12-31
           Previous version ended 2021-02-20 → only 519 rows.
           5 years gives ~1250 trading days per index.

Downloads:
  ^NSEI  = NIFTY 50 Index (Indian market)
  ^GSPC  = S&P 500 Index  (US market)
"""

import yfinance as yf
import pandas as pd

def fetch_stocks():
    START = "2019-01-01"
    END   = "2024-12-31"

    for name, ticker, outfile in [
        ("NIFTY 50",  "^NSEI",  "data/nifty.csv"),
        ("S&P 500",   "^GSPC",  "data/sp500.csv"),
    ]:
        print(f"Downloading {name} ({ticker}) from {START} to {END} ...")
        df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

        if df.empty:
            print(f"  ERROR: No data returned for {ticker}. Check internet connection.")
            continue

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={"date": "time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"]).dt.normalize()

        df = df[["time", "open", "high", "low", "close", "volume"]]
        df.to_csv(outfile, index=False)
        print(f"  Saved {len(df)} rows  ({df['time'].min().date()} → {df['time'].max().date()})  →  {outfile}")

if __name__ == "__main__":
    fetch_stocks()