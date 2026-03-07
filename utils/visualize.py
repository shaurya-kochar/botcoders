"""
Visualization utilities for model evaluation results.

All functions save plots to `output_dir` and return the saved file paths.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — safe for scripts & notebooks
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── colour palette ───────────────────────────────────────────────────────────
COLORS = {
    "no_sent":   "#4C72B0",   # blue
    "with_sent": "#DD8452",   # orange
}
METRIC_LABELS = {"rmse": "RMSE (↓ better)", "mae": "MAE (↓ better)", "r2": "R² (↑ better)"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _savefig(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


# ── 1. Comparative metric bar charts ─────────────────────────────────────────

def plot_metrics_comparison(all_results: dict, output_dir: str) -> list[str]:
    """
    Grouped bar charts comparing RMSE / MAE / R² across all models,
    split: with-sentiment vs without-sentiment, train vs test.

    all_results layout:
        {
          "Linear Regression": {"no_sent": results_dict, "with_sent": results_dict},
          "Random Forest":     {"no_sent": ..., "with_sent": ...},
          "LSTM":              {"no_sent": ..., "with_sent": ...},
        }
    """
    saved = []
    model_names = list(all_results.keys())
    x = np.arange(len(model_names))
    width = 0.35

    for split in ("train", "test"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Model Comparison — {split.capitalize()} Metrics", fontsize=14, fontweight="bold")

        for ax, (metric, ylabel) in zip(axes, METRIC_LABELS.items()):
            vals_no   = [all_results[m]["no_sent"][split][metric]   for m in model_names]
            vals_with = [all_results[m]["with_sent"][split][metric] for m in model_names]

            bars1 = ax.bar(x - width/2, vals_no,   width, label="Without Sentiment", color=COLORS["no_sent"],   alpha=0.85)
            bars2 = ax.bar(x + width/2, vals_with, width, label="With Sentiment",    color=COLORS["with_sent"], alpha=0.85)

            ax.set_title(ylabel, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=15, ha="right")
            ax.legend(fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

            for bar in (*bars1, *bars2):
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        path = os.path.join(output_dir, f"metrics_comparison_{split}.png")
        saved.append(_savefig(fig, path))

    return saved


# ── 2. Actual vs Predicted ────────────────────────────────────────────────────

def plot_actual_vs_predicted(all_results: dict, output_dir: str) -> list[str]:
    """
    One subplot per model×sentiment showing actual vs predicted on the test set.
    """
    model_names = list(all_results.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models))
    fig.suptitle("Actual vs Predicted — Test Set", fontsize=14, fontweight="bold")

    for row, model_name in enumerate(model_names):
        for col, (sent_key, sent_label) in enumerate(
            [("no_sent", "Without Sentiment"), ("with_sent", "With Sentiment")]
        ):
            ax = axes[row][col]
            res = all_results[model_name][sent_key]["test"]
            preds, targets = res["preds"], res["targets"]

            ax.plot(targets, label="Actual",    color="#2ca02c", linewidth=1.5)
            ax.plot(preds,   label="Predicted", color="#d62728", linewidth=1.5, linestyle="--")
            ax.set_title(f"{model_name} | {sent_label}", fontsize=10)
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Close Price")
            ax.legend(fontsize=8)

            r2 = res["r2"]
            ax.text(0.02, 0.95, f"R²={r2:.3f}", transform=ax.transAxes,
                    fontsize=9, va="top", color="#333333",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    plt.tight_layout()
    path = os.path.join(output_dir, "actual_vs_predicted.png")
    return [_savefig(fig, path)]


# ── 3. LSTM training & validation loss curves ─────────────────────────────────

def plot_lstm_loss(all_results: dict, output_dir: str) -> list[str]:
    """
    Training vs validation loss curves for both LSTM variants.
    """
    lstm_results = all_results.get("LSTM", {})
    if not lstm_results:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("LSTM — Training vs Validation Loss", fontsize=14, fontweight="bold")

    for ax, (sent_key, sent_label) in zip(
        axes, [("no_sent", "Without Sentiment"), ("with_sent", "With Sentiment")]
    ):
        res = lstm_results[sent_key]
        train_loss = res.get("train_loss_history", [])
        val_loss   = res.get("val_loss_history",   [])
        epochs = range(1, len(train_loss) + 1)

        ax.plot(epochs, train_loss, label="Train Loss",      color=COLORS["no_sent"],   linewidth=1.5)
        ax.plot(epochs, val_loss,   label="Validation Loss", color=COLORS["with_sent"], linewidth=1.5, linestyle="--")
        ax.set_title(sent_label, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "lstm_loss_curves.png")
    return [_savefig(fig, path)]


# ── 4. Sentiment impact (delta bars) ─────────────────────────────────────────

def plot_sentiment_impact(all_results: dict, output_dir: str) -> list[str]:
    """
    Bar chart showing the change in each metric when sentiment features are added.
    Δmetric = metric_with_sentiment − metric_without_sentiment (test split only).
    For RMSE/MAE a negative Δ means improvement; for R² a positive Δ means improvement.
    """
    model_names = list(all_results.keys())
    x = np.arange(len(model_names))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Sentiment Impact on Test Metrics (Δ = with − without)", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, ["rmse", "mae", "r2"]):
        deltas = [
            all_results[m]["with_sent"]["test"][metric] -
            all_results[m]["no_sent"]["test"][metric]
            for m in model_names
        ]
        bar_colors = ["#2ca02c" if (metric == "r2" and d > 0) or (metric != "r2" and d < 0)
                      else "#d62728" for d in deltas]
        bars = ax.bar(x, deltas, width=0.5, color=bar_colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        for bar, d in zip(bars, deltas):
            ax.annotate(f"{d:+.3f}", xy=(bar.get_x() + bar.get_width()/2, d),
                        xytext=(0, 3 if d >= 0 else -12), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "sentiment_impact.png")
    return [_savefig(fig, path)]


# ── master function ───────────────────────────────────────────────────────────

def generate_all_plots(all_results: dict, output_dir: str) -> list[str]:
    """Run all plot functions and return a list of saved file paths."""
    paths = []
    paths += plot_metrics_comparison(all_results, output_dir)
    paths += plot_actual_vs_predicted(all_results, output_dir)
    paths += plot_lstm_loss(all_results, output_dir)
    paths += plot_sentiment_impact(all_results, output_dir)
    return paths
