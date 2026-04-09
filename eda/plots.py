"""
eda_plots.py
=============
Generates 4 plots from your final_nifty.csv and final_sp500.csv.
Run AFTER merge_sequence.py.

  plots/price_trend.png           — NIFTY vs SP500 normalised close
  plots/daily_sentiment_trend.png — S_t over time for both markets
  plots/sentiment_histogram.png   — Distribution of S_t by market
  plots/correlation_plot.png      — Feature correlation heatmap
  outputs/eda_report.txt          — Text summary of the dataset
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

os.makedirs("plots",   exist_ok=True)
os.makedirs("outputs", exist_ok=True)


def load():
    nifty = pd.read_csv("data/final_nifty.csv")
    sp500 = pd.read_csv("data/final_sp500.csv")
    nifty["time"] = pd.to_datetime(nifty["bucket"])
    sp500["time"] = pd.to_datetime(sp500["bucket"])
    print(f"NIFTY: {len(nifty)} rows  |  SP500: {len(sp500)} rows")
    return nifty, sp500


# ── Plot 1: Price trends ──────────────────────────────────────
def plot_price(nifty, sp500):
    print("Generating price_trend.png ...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=False)
    fig.suptitle("Normalised Closing Prices (Min-Max) — 2019–2024", fontsize=13, fontweight="bold")

    ax1.plot(nifty["time"], nifty["close"], color="#2563EB", lw=1.4, label="NIFTY 50")
    ax1.set_title("NIFTY 50  (Normalised 0–1)", fontweight="bold")
    ax1.set_ylabel("Close Price (normalised)")
    ax1.grid(alpha=0.2); ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax2.plot(sp500["time"], sp500["close"], color="#d97706", lw=1.4, label="S&P 500")
    ax2.set_title("S&P 500  (Normalised 0–1)", fontweight="bold")
    ax2.set_ylabel("Close Price (normalised)")
    ax2.grid(alpha=0.2); ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig("plots/price_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → plots/price_trend.png")


# ── Plot 2: Sentiment over time ───────────────────────────────
def plot_sentiment(nifty, sp500):
    print("Generating daily_sentiment_trend.png ...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=False)
    fig.suptitle("Daily Weighted Sentiment S_t — Both Markets", fontsize=13, fontweight="bold")

    for ax, df, label, color in [
        (ax1, nifty, "NIFTY 50", "#2563EB"),
        (ax2, sp500, "S&P 500",  "#d97706"),
    ]:
        ax.plot(df["time"], df["S_t"], color=color, lw=1.0, alpha=0.9)
        ax.fill_between(df["time"], df["S_t"], 0,
                        where=df["S_t"] >= 0,  alpha=0.15, color="green")
        ax.fill_between(df["time"], df["S_t"], 0,
                        where=df["S_t"] < 0,   alpha=0.15, color="red")
        ax.axhline( 0.05, color="green", linestyle="--", lw=0.8, alpha=0.6)
        ax.axhline(-0.05, color="red",   linestyle="--", lw=0.8, alpha=0.6)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_title(f"{label} — S_t  (market-specific sentiment)",  fontweight="bold")
        ax.set_ylabel("S_t  (−1=Bearish, +1=Bullish)")
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig("plots/daily_sentiment_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → plots/daily_sentiment_trend.png")


# ── Plot 3: Sentiment distribution ───────────────────────────
def plot_histogram(nifty, sp500):
    print("Generating sentiment_histogram.png ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribution of Daily S_t", fontsize=13, fontweight="bold")

    for ax, df, label, color in [
        (ax1, nifty, "NIFTY 50", "#2563EB"),
        (ax2, sp500, "S&P 500",  "#d97706"),
    ]:
        ax.hist(df["S_t"], bins=50, color=color, edgecolor="white", alpha=0.8)
        ax.axvline( 0.05, color="green", linestyle="--", lw=1.5)
        ax.axvline(-0.05, color="red",   linestyle="--", lw=1.5)
        ax.axvline(df["S_t"].mean(), color="black", lw=2,
                   label=f"Mean={df['S_t'].mean():.3f}")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("S_t score")
        ax.set_ylabel("Days")
        ax.legend(); ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("plots/sentiment_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → plots/sentiment_histogram.png")


# ── Plot 4: Correlation heatmap ───────────────────────────────
def plot_correlation(nifty, sp500):
    print("Generating correlation_plot.png ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("Feature Correlation Heatmap", fontsize=13, fontweight="bold")

    lag_cols  = [f"S_t-{i}" for i in range(1, 7)]
    feat_cols = (["S_t"] + lag_cols + ["shock","momentum","confidence"]
                 + ["open","high","low","close","volume","daily_return"]
                 + ["rsi","macd","bb_width","ema_20","ema_50","label_int"])

    for ax, df, label in [(ax1, nifty, "NIFTY 50"), (ax2, sp500, "S&P 500")]:
        cols = [c for c in feat_cols if c in df.columns]
        corr = df[cols].corr()
        im   = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cols, fontsize=7)
        for i in range(len(corr)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                        ha="center", va="center", fontsize=5.5)
        ax.set_title(label, fontweight="bold")

    plt.tight_layout()
    plt.savefig("plots/correlation_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → plots/correlation_plot.png")


# ── EDA Report ───────────────────────────────────────────────
def write_report(nifty, sp500):
    print("Writing outputs/eda_report.txt ...")
    lines = [
        "=" * 58,
        "  EDA REPORT — Financial Sentiment LSTM Dataset",
        "=" * 58,
        f"\n  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    for name, df in [("NIFTY 50", nifty), ("S&P 500", sp500)]:
        lines += [
            f"\n{'─'*58}",
            f"  {name}",
            f"{'─'*58}",
            f"  File rows   : {len(df):,}",
            f"  Columns     : {len(df.columns)}",
            f"  Date range  : {df['time'].min().date()} → {df['time'].max().date()}",
            f"  Labels      : {df['label'].value_counts().to_dict()}",
            f"\n  S_t stats:",
            f"    mean={df['S_t'].mean():.4f}  std={df['S_t'].std():.4f}  "
            f"min={df['S_t'].min():.4f}  max={df['S_t'].max():.4f}",
            f"\n  Close (normalised) stats:",
            f"    mean={df['close'].mean():.4f}  std={df['close'].std():.4f}",
            f"\n  RSI stats:",
            f"    mean={df['rsi'].mean():.2f}  std={df['rsi'].std():.2f}" if "rsi" in df.columns else "  RSI: not present",
        ]

    lines += [
        f"\n{'─'*58}",
        "  LSTM READINESS",
        f"{'─'*58}",
        "  Sequence length   : 6 time steps",
        f"  Features per step : {len([c for c in nifty.columns if c not in ['bucket','time','label','ticker','company','index_group']])}",
        f"  NIFTY sequences   : ~{len(nifty)-6}",
        f"  SP500 sequences   : ~{len(sp500)-6}",
        f"  Input shape       : X = (N, 6, num_features)",
        "\n" + "=" * 58,
    ]

    report = "\n".join(lines)
    with open("outputs/eda_report.txt", "w") as f:
        f.write(report)
    print(report)


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("eda_plots.py — Generating all plots + report")
    nifty, sp500 = load()
    plot_price(nifty, sp500)
    plot_sentiment(nifty, sp500)
    plot_histogram(nifty, sp500)
    plot_correlation(nifty, sp500)
    write_report(nifty, sp500)
    print("\nAll done! Check plots/ and outputs/ folders.")