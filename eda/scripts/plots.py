"""
plots.py
=========
Generates 6 plots from your final CSV files.
Run after merge_sequence.py.

  plots/01_price_trend.png         — NIFTY vs SP500 closing prices
  plots/02_sentiment_trend.png     — Daily S_t over time
  plots/03_sentiment_vs_price.png  — Sentiment and price on same chart
  plots/04_label_distribution.png  — Bullish/Neutral/Bearish pie charts
  plots/05_correlation_heatmap.png — Feature correlations
  plots/06_rsi_over_time.png       — RSI with overbought/oversold lines
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

os.makedirs("plots", exist_ok=True)

def load():
    nifty = pd.read_csv("data/final_nifty.csv")
    sp500 = pd.read_csv("data/final_sp500.csv")
    for df in [nifty, sp500]:
        dc = "bucket" if "bucket" in df.columns else "time"
        df["time"] = pd.to_datetime(df[dc])
    return nifty.sort_values("time"), sp500.sort_values("time")


# ── Plot 1: Price Trend ───────────────────────────────────────────────────────
def plot_price(nifty, sp500):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Normalised Closing Prices  2019–2024", fontsize=13, fontweight="bold")
    a1.plot(nifty["time"], nifty["close"], color="#2563EB", lw=1.4, label="NIFTY 50")
    a1.set_title("NIFTY 50", fontweight="bold"); a1.set_ylabel("Close (0-1)")
    a1.grid(alpha=0.2); a1.legend(); a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    a2.plot(sp500["time"], sp500["close"], color="#d97706", lw=1.4, label="S&P 500")
    a2.set_title("S&P 500", fontweight="bold"); a2.set_ylabel("Close (0-1)")
    a2.grid(alpha=0.2); a2.legend(); a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig("plots/01_price_trend.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved plots/01_price_trend.png")


# ── Plot 2: Sentiment Trend ───────────────────────────────────────────────────
def plot_sentiment(nifty, sp500):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Daily Sentiment Score S_t  2019–2024", fontsize=13, fontweight="bold")
    for ax, df, label, col in [(a1, nifty, "NIFTY 50", "#2563EB"), (a2, sp500, "S&P 500", "#d97706")]:
        ax.plot(df["time"], df["S_t"], color=col, lw=1.0, alpha=0.9)
        ax.fill_between(df["time"], df["S_t"], 0, where=df["S_t"]>=0, alpha=0.15, color="green")
        ax.fill_between(df["time"], df["S_t"], 0, where=df["S_t"]<0,  alpha=0.15, color="red")
        ax.axhline(0.05, color="green", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(-0.05, color="red", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_title(label, fontweight="bold"); ax.set_ylabel("S_t (-1 to +1)")
        ax.grid(alpha=0.2); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig("plots/02_sentiment_trend.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved plots/02_sentiment_trend.png")


# ── Plot 3: Sentiment + Price Together ───────────────────────────────────────
def plot_sent_vs_price(nifty, sp500):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Sentiment vs Price  (dual axis)", fontsize=13, fontweight="bold")
    for ax, df, label, c1, c2 in [
        (axes[0], nifty, "NIFTY 50", "#2563EB", "#f59e0b"),
        (axes[1], sp500, "S&P 500",  "#d97706", "#7c3aed"),
    ]:
        ax2 = ax.twinx()
        ax.plot(df["time"], df["close"], color=c1, lw=1.4, label="Close price")
        ax2.plot(df["time"], df["S_t"], color=c2, lw=0.9, alpha=0.7, label="S_t sentiment")
        ax2.axhline(0, color="gray", lw=0.4)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Close price (normalised)", color=c1)
        ax2.set_ylabel("Sentiment S_t", color=c2)
        ax.grid(alpha=0.15); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        lines1, lbls1 = ax.get_legend_handles_labels()
        lines2, lbls2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, lbls1+lbls2, fontsize=8)
    plt.tight_layout()
    plt.savefig("plots/03_sentiment_vs_price.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved plots/03_sentiment_vs_price.png")


# ── Plot 4: Label Distribution ────────────────────────────────────────────────
def plot_labels(nifty, sp500):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sentiment Label Distribution", fontsize=13, fontweight="bold")
    colors = {"Bullish": "#16a34a", "Neutral": "#64748b", "Bearish": "#dc2626"}
    for ax, df, title in [(a1, nifty, "NIFTY 50"), (a2, sp500, "S&P 500")]:
        counts = df["label"].value_counts()
        clrs = [colors.get(l, "#888") for l in counts.index]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index,
            autopct="%1.1f%%", colors=clrs,
            textprops={"fontsize": 11}
        )
        ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/04_label_distribution.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved plots/04_label_distribution.png")


# ── Plot 5: Correlation Heatmap ───────────────────────────────────────────────
def plot_correlation(nifty, sp500):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    cols = ["S_t","open","high","low","close","volume","daily_return",
            "rsi","macd","bb_width","momentum","confidence","label_int"]
    for ax, df, title in [(a1, nifty, "NIFTY 50"), (a2, sp500, "S&P 500")]:
        c = [x for x in cols if x in df.columns]
        corr = df[c].corr()
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_xticks(range(len(c))); ax.set_yticks(range(len(c)))
        ax.set_xticklabels(c, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(c, fontsize=8)
        for i in range(len(corr)):
            for j in range(len(c)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/05_correlation_heatmap.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved plots/05_correlation_heatmap.png")


# ── Plot 6: RSI ───────────────────────────────────────────────────────────────
def plot_rsi(nifty, sp500):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("RSI(14)  —  Overbought > 70, Oversold < 30", fontsize=13, fontweight="bold")
    for ax, df, label, col in [(a1, nifty, "NIFTY 50", "#2563EB"), (a2, sp500, "S&P 500", "#d97706")]:
        if "rsi" not in df.columns: continue
        ax.plot(df["time"], df["rsi"], color=col, lw=1.0)
        ax.axhline(70, color="red",   ls="--", lw=1.2, label="Overbought (70)")
        ax.axhline(30, color="green", ls="--", lw=1.2, label="Oversold (30)")
        ax.fill_between(df["time"], df["rsi"], 70, where=df["rsi"]>=70, alpha=0.2, color="red")
        ax.fill_between(df["time"], df["rsi"], 30, where=df["rsi"]<=30, alpha=0.2, color="green")
        ax.set_title(label, fontweight="bold"); ax.set_ylabel("RSI")
        ax.set_ylim(0, 100); ax.grid(alpha=0.2); ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig("plots/06_rsi_over_time.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Saved plots/06_rsi_over_time.png")


if __name__ == "__main__":
    print("plots.py — Generating all 6 plots")
    nifty, sp500 = load()
    print(f"  NIFTY rows: {len(nifty)} | SP500 rows: {len(sp500)}")
    plot_price(nifty, sp500)
    plot_sentiment(nifty, sp500)
    plot_sent_vs_price(nifty, sp500)
    plot_labels(nifty, sp500)
    plot_correlation(nifty, sp500)
    plot_rsi(nifty, sp500)
    print("\nAll 6 plots saved in plots/ folder!")