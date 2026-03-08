"""
exploratory_analysis.py
───────────────────────
Reads the raw sentiment & price CSVs and produces:

  1. A printed (and saved) EDA report covering
       • sentiment distribution stats
       • text length distribution stats
       • dataset imbalance across sentiment classes & sources
  2. Four publication-quality plots saved into eda/plots/
       • sentiment_histogram.png
       • daily_sentiment_trend.png
       • price_trend.png
       • correlation_plot.png

All heavy lifting is done with pandas + matplotlib + seaborn.
"""

import sys
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for any environment)
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    RAW_SENTIMENT_CSV, RAW_PRICE_CSV, ALL_SYMBOLS,
    PLOTS_DIR, EDA_REPORT_TXT,
    PLOT_STYLE, PLOT_DPI, FIG_SIZE, PALETTE,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Load data ─────────────────────────────────────────────────────────────────

def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSVs; abort with a helpful message if they don't exist."""
    for path, label in [(RAW_SENTIMENT_CSV, "sentiment"), (RAW_PRICE_CSV, "price")]:
        if not path.exists():
            sys.exit(
                f"[ERROR] {label} CSV not found at {path}.\n"
                "       Run  python generate_data.py  first."
            )

    sent = pd.read_csv(RAW_SENTIMENT_CSV, parse_dates=["date"])
    price = pd.read_csv(RAW_PRICE_CSV, parse_dates=["date"])
    return sent, price


# ─── 1. Exploratory statistics ─────────────────────────────────────────────────

def _bucket_label(score: float) -> str:
    """Map a continuous sentiment score to a human-readable class."""
    if score > 0.25:
        return "Positive"
    elif score < -0.25:
        return "Negative"
    else:
        return "Neutral"


def compute_eda_stats(sent: pd.DataFrame) -> str:
    """Return a plain-text EDA report string."""
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("  EXPLORATORY DATA ANALYSIS REPORT")
    lines.append("=" * 72)

    # --- Sentiment distribution ---
    lines.append("\n1. SENTIMENT DISTRIBUTION")
    lines.append("-" * 40)
    desc = sent["sentiment_score"].describe()
    for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        lines.append(f"   {stat:>6s}:  {desc[stat]:>10.4f}")

    skew = sent["sentiment_score"].skew()
    kurt = sent["sentiment_score"].kurtosis()
    lines.append(f"   {'skew':>6s}:  {skew:>10.4f}")
    lines.append(f"   {'kurt':>6s}:  {kurt:>10.4f}")

    # --- Text length distribution ---
    lines.append("\n2. TEXT LENGTH DISTRIBUTION (characters)")
    lines.append("-" * 40)
    sent["text_length"] = sent["text"].str.len()
    tl = sent["text_length"].describe()
    for stat in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        lines.append(f"   {stat:>6s}:  {tl[stat]:>10.1f}")

    # word-level stats too
    sent["word_count"] = sent["text"].str.split().str.len()
    wc = sent["word_count"].describe()
    lines.append("\n   Word count:")
    for stat in ["mean", "std", "min", "max"]:
        lines.append(f"     {stat:>6s}:  {wc[stat]:>10.1f}")

    # --- Dataset imbalance ---
    lines.append("\n3. DATASET IMBALANCE")
    lines.append("-" * 40)

    sent["sentiment_class"] = sent["sentiment_score"].apply(_bucket_label)
    class_counts = sent["sentiment_class"].value_counts()
    total = len(sent)
    lines.append("   Sentiment class distribution:")
    for cls, cnt in class_counts.items():
        pct = cnt / total * 100
        lines.append(f"     {cls:<10s}  {cnt:>7,}  ({pct:5.1f}%)")

    imbalance_ratio = class_counts.max() / class_counts.min()
    lines.append(f"   Imbalance ratio (max/min): {imbalance_ratio:.2f}")

    lines.append("\n   Records per source:")
    src_counts = sent["source"].value_counts()
    for src, cnt in src_counts.items():
        pct = cnt / total * 100
        lines.append(f"     {src:<14s}  {cnt:>7,}  ({pct:5.1f}%)")

    lines.append("\n   Records per symbol:")
    sym_counts = sent["symbol"].value_counts().sort_index()
    for sym, cnt in sym_counts.items():
        pct = cnt / total * 100
        lines.append(f"     {sym:<10s}  {cnt:>7,}  ({pct:5.1f}%)")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


# ─── 2. Visualizations ────────────────────────────────────────────────────────

def _apply_style():
    """Apply a consistent plot style."""
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        # fallback if the named style isn't available
        plt.style.use("ggplot")
    sns.set_palette(PALETTE)


def plot_sentiment_histogram(sent: pd.DataFrame) -> None:
    """Histogram of sentiment_score with KDE, coloured by sentiment class."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # overall distribution
    ax = axes[0]
    ax.hist(
        sent["sentiment_score"], bins=50, color="#4C72B0",
        edgecolor="white", alpha=0.85, density=True,
    )
    sent["sentiment_score"].plot.kde(ax=ax, color="crimson", linewidth=2)
    ax.set_title("Overall Sentiment Score Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Density")
    ax.axvline(0, color="grey", linestyle="--", alpha=0.6)

    # per-class bar chart
    ax = axes[1]
    if "sentiment_class" not in sent.columns:
        sent["sentiment_class"] = sent["sentiment_score"].apply(_bucket_label)
    order = ["Negative", "Neutral", "Positive"]
    colors = {"Negative": "#d9534f", "Neutral": "#f0ad4e", "Positive": "#5cb85c"}
    class_counts = sent["sentiment_class"].value_counts().reindex(order, fill_value=0)
    bars = ax.bar(class_counts.index, class_counts.values,
                  color=[colors[c] for c in class_counts.index], edgecolor="white")
    for bar, val in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val:,}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Sentiment Class Counts", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")

    fig.tight_layout()
    path = PLOTS_DIR / "sentiment_histogram.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_daily_sentiment_trend(sent: pd.DataFrame) -> None:
    """Line chart of daily average sentiment score per symbol."""
    _apply_style()
    daily = sent.groupby(["date", "symbol"])["sentiment_score"].mean().reset_index()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    symbols = sorted(daily["symbol"].unique())
    cmap = plt.cm.get_cmap("tab10", len(symbols))

    for idx, sym in enumerate(symbols):
        sub = daily[daily["symbol"] == sym].sort_values("date")
        ax.plot(sub["date"], sub["sentiment_score"],
                label=sym, linewidth=1.4, alpha=0.85, color=cmap(idx))

    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.set_title("Daily Average Sentiment Score by Symbol", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Sentiment Score")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = PLOTS_DIR / "daily_sentiment_trend.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_price_trend(price: pd.DataFrame) -> None:
    """Closing-price trend with a 7-day moving average for each symbol."""
    _apply_style()
    # normalise prices to % change from day-0 so all symbols fit one chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # — raw close prices (log scale)
    ax = axes[0]
    symbols = sorted(price["symbol"].unique())
    cmap = plt.cm.get_cmap("tab10", len(symbols))
    for idx, sym in enumerate(symbols):
        sub = price[price["symbol"] == sym].sort_values("date")
        ax.plot(sub["date"], sub["close"], label=sym, linewidth=1.3, color=cmap(idx))
    ax.set_title("Daily Closing Price", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (log)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.tick_params(axis="x", rotation=30)

    # — normalised % change from first day
    ax = axes[1]
    for idx, sym in enumerate(symbols):
        sub = price[price["symbol"] == sym].sort_values("date").copy()
        first_close = sub["close"].iloc[0]
        if first_close == 0:
            continue
        sub["pct_change"] = (sub["close"] / first_close - 1) * 100
        ax.plot(sub["date"], sub["pct_change"], label=sym, linewidth=1.3, color=cmap(idx))
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
    ax.set_title("Normalised Price Change (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("% Change from Day 1")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    path = PLOTS_DIR / "price_trend.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_correlation(sent: pd.DataFrame, price: pd.DataFrame) -> None:
    """
    Heatmap showing Pearson correlation between daily sentiment metrics and
    price metrics across all symbols pooled.
    """
    _apply_style()

    # aggregate sentiment per (date, symbol)
    daily_sent = (
        sent.groupby(["date", "symbol"])
        .agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            positive_mean=("positive", "mean"),
            negative_mean=("negative", "mean"),
            neutral_mean=("neutral", "mean"),
        )
        .reset_index()
    )

    merged = pd.merge(daily_sent, price, on=["date", "symbol"], how="inner")

    cols = [
        "sentiment_mean", "sentiment_std", "positive_mean",
        "negative_mean", "neutral_mean",
        "close", "volume", "sentiment_overlay",
    ]
    corr = merged[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, linewidths=0.5, square=True, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation: Sentiment vs. Price Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = PLOTS_DIR / "correlation_plot.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    sent, price = _load()

    # 1. EDA report
    print("\n── Exploratory Statistics ──")
    report = compute_eda_stats(sent)
    print(report)
    EDA_REPORT_TXT.write_text(report, encoding="utf-8")
    print(f"\n  Report saved → {EDA_REPORT_TXT}")

    # 2. Visualizations
    print("\n── Generating Plots ──")
    plot_sentiment_histogram(sent)
    plot_daily_sentiment_trend(sent)
    plot_price_trend(price)
    plot_correlation(sent, price)

    print("\nExploratory analysis complete ✓")


if __name__ == "__main__":
    main()
