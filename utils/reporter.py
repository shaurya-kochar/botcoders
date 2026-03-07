"""
Auto-generates / updates report.md with evaluation metrics and embedded plots.
"""
import os
from datetime import datetime


def _metric_row(model_name: str, sent_label: str, split: str, res: dict) -> str:
    m = res[split]
    return f"| {model_name} | {sent_label} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['r2']:.4f} |"


def _rel_path(abs_path: str, report_dir: str) -> str:
    """Return a path relative to the report file for use in markdown image links."""
    return os.path.relpath(abs_path, report_dir)


def generate_report(
    all_results: dict,
    plot_paths: list[str],
    report_path: str,
) -> str:
    """
    Write (or overwrite) report_path with a full markdown evaluation report.

    Parameters
    ----------
    all_results : dict
        Nested dict: {model_name: {sent_key: results_dict, ...}, ...}
        where sent_key is "no_sent" or "with_sent".
    plot_paths  : list[str]
        Absolute paths to saved plot PNG files.
    report_path : str
        Absolute path where report.md should be written.
    """
    report_dir = os.path.dirname(report_path)
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_names = list(all_results.keys())

    lines = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "# Model Evaluation Report",
        "",
        f"> Auto-generated on **{timestamp}**",
        "",
        "## Overview",
        "",
        "This report compares **Linear Regression**, **Random Forest**, and **LSTM** on a stock",
        "close-price prediction task, evaluated with and without Twitter sentiment features.",
        "",
        "**Features without sentiment (6):** open, high, low, volume, running_max, running_min  ",
        "**Additional sentiment features (2):** positiveness, negativeness  ",
        "**Target:** close price  ",
        "",
    ]

    # ── Train metrics table ──────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Training Set Metrics",
        "",
        "| Model | Sentiment | RMSE ↓ | MAE ↓ | R² ↑ |",
        "|---|---|---|---|---|",
    ]
    for m in model_names:
        lines.append(_metric_row(m, "Without", "train", all_results[m]["no_sent"]))
        lines.append(_metric_row(m, "With",    "train", all_results[m]["with_sent"]))
    lines.append("")

    # ── Test metrics table ───────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Test Set Metrics (Validation)",
        "",
        "| Model | Sentiment | RMSE ↓ | MAE ↓ | R² ↑ |",
        "|---|---|---|---|---|",
    ]
    for m in model_names:
        lines.append(_metric_row(m, "Without", "test", all_results[m]["no_sent"]))
        lines.append(_metric_row(m, "With",    "test", all_results[m]["with_sent"]))
    lines.append("")

    # ── Sentiment impact summary ─────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Sentiment Feature Impact (Test Set)",
        "",
        "Δ = metric_with_sentiment − metric_without_sentiment  ",
        "For RMSE/MAE: negative Δ = improvement. For R²: positive Δ = improvement.",
        "",
        "| Model | ΔRMSE | ΔMAE | ΔR² |",
        "|---|---|---|---|",
    ]
    for m in model_names:
        no  = all_results[m]["no_sent"]["test"]
        yes = all_results[m]["with_sent"]["test"]
        d_rmse = yes["rmse"] - no["rmse"]
        d_mae  = yes["mae"]  - no["mae"]
        d_r2   = yes["r2"]   - no["r2"]
        arrow_rmse = "✅" if d_rmse < 0 else "❌"
        arrow_mae  = "✅" if d_mae  < 0 else "❌"
        arrow_r2   = "✅" if d_r2   > 0 else "❌"
        lines.append(
            f"| {m} | {d_rmse:+.4f} {arrow_rmse} | {d_mae:+.4f} {arrow_mae} | {d_r2:+.4f} {arrow_r2} |"
        )
    lines.append("")

    # ── LSTM loss history ────────────────────────────────────────────────────
    if "LSTM" in all_results:
        lines += [
            "---",
            "",
            "## LSTM Loss History (final epoch)",
            "",
            "| Variant | Final Train Loss | Final Val Loss |",
            "|---|---|---|",
        ]
        for sent_key, label in [("no_sent", "Without Sentiment"), ("with_sent", "With Sentiment")]:
            lstm_res = all_results["LSTM"][sent_key]
            tl = lstm_res.get("train_loss_history", [])
            vl = lstm_res.get("val_loss_history", [])
            final_train = tl[-1] if tl else float("nan")
            final_val   = vl[-1] if vl else float("nan")
            lines.append(f"| {label} | {final_train:.6f} | {final_val:.6f} |")
        lines.append("")

    # ── Plots ────────────────────────────────────────────────────────────────
    PLOT_TITLES = {
        "metrics_comparison_train.png":  "Training Metrics Comparison",
        "metrics_comparison_test.png":   "Test Metrics Comparison",
        "actual_vs_predicted.png":       "Actual vs Predicted (Test Set)",
        "lstm_loss_curves.png":          "LSTM Training & Validation Loss",
        "sentiment_impact.png":          "Sentiment Feature Impact",
    }

    if plot_paths:
        lines += ["---", "", "## Plots", ""]
        for abs_path in plot_paths:
            fname = os.path.basename(abs_path)
            title = PLOT_TITLES.get(fname, fname)
            rel   = _rel_path(abs_path, report_dir)
            lines += [f"### {title}", "", f"![{title}]({rel})", ""]

    # ── Write file ───────────────────────────────────────────────────────────
    content = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  Report saved → {report_path}")
    return report_path
