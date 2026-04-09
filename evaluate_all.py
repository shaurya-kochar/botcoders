"""
evaluate_all.py — Master evaluation script.

Runs Linear Regression, Random Forest, and LSTM (with & without sentiment),
generates all visualisation plots, and auto-updates report.md.

Usage:
    python evaluate_all.py
    python evaluate_all.py --csv ~/dataset/inlp_project/final_sub/combined_clean.csv
    python evaluate_all.py --csv <path> --report report.md --plots plots/ --checkpoints checkpoints/
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.linear_regression import run   as lr_run
from models.random_forest      import run   as rf_run
from models.lstm               import train as lstm_train
from utils.visualize           import generate_all_plots
from utils.reporter            import generate_report


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CSV         = os.path.expanduser("~/dataset/inlp_project/final_sub/combined_clean.csv")
DEFAULT_REPORT      = os.path.join(ROOT, "report.md")
DEFAULT_PLOTS       = os.path.join(ROOT, "plots")
DEFAULT_CHECKPOINTS = os.path.join(ROOT, "checkpoints")
LSTM_EPOCHS         = 50
LSTM_WINDOW         = 5


# ── Runner helpers ────────────────────────────────────────────────────────────

def run_linear_regression(csv_path: str, checkpoint_dir: str) -> dict:
    print("\n" + "=" * 60)
    print("  LINEAR REGRESSION")
    print("=" * 60)
    _, res_no   = lr_run(csv_path, with_sentiment=False, checkpoint_dir=checkpoint_dir)
    _, res_with = lr_run(csv_path, with_sentiment=True,  checkpoint_dir=checkpoint_dir)
    return {"no_sent": res_no, "with_sent": res_with}


def run_random_forest(csv_path: str, checkpoint_dir: str) -> dict:
    print("\n" + "=" * 60)
    print("  RANDOM FOREST")
    print("=" * 60)
    _, res_no   = rf_run(csv_path, with_sentiment=False, checkpoint_dir=checkpoint_dir)
    _, res_with = rf_run(csv_path, with_sentiment=True,  checkpoint_dir=checkpoint_dir)
    return {"no_sent": res_no, "with_sent": res_with}


def run_lstm(csv_path: str, epochs: int = LSTM_EPOCHS, checkpoint_dir: str = None) -> dict:
    print("\n" + "=" * 60)
    print("  LSTM")
    print("=" * 60)
    _, res_no   = lstm_train(csv_path, with_sentiment=False, epochs=epochs,
                             window_size=LSTM_WINDOW, checkpoint_dir=checkpoint_dir)
    _, res_with = lstm_train(csv_path, with_sentiment=True,  epochs=epochs,
                             window_size=LSTM_WINDOW, checkpoint_dir=checkpoint_dir)
    return {"no_sent": res_no, "with_sent": res_with}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate all models and generate report")
    parser.add_argument("--csv",         default=DEFAULT_CSV,         help="Path to dataset CSV")
    parser.add_argument("--report",      default=DEFAULT_REPORT,      help="Output path for report.md")
    parser.add_argument("--plots",       default=DEFAULT_PLOTS,       help="Directory to save plot PNGs")
    parser.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS, help="Directory to save model checkpoints")
    parser.add_argument("--epochs",      default=LSTM_EPOCHS, type=int, help="LSTM training epochs")
    args = parser.parse_args()

    print(f"\nDataset     : {args.csv}")
    print(f"Report      : {args.report}")
    print(f"Plots       : {args.plots}")
    print(f"Checkpoints : {args.checkpoints}")

    # ── 1. Train & evaluate all models ────────────────────────────────────
    all_results = {
        "Linear Regression": run_linear_regression(args.csv, args.checkpoints),
        "Random Forest":     run_random_forest(args.csv, args.checkpoints),
        "LSTM":              run_lstm(args.csv, epochs=args.epochs, checkpoint_dir=args.checkpoints),
    }

    # ── 2. Generate plots ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GENERATING PLOTS")
    print("=" * 60)
    plot_paths = generate_all_plots(all_results, args.plots)
    for p in plot_paths:
        print(f"  Saved: {p}")

    # ── 3. Write report ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  WRITING REPORT")
    print("=" * 60)
    generate_report(all_results, plot_paths, args.report)

    print("\nDone.")


if __name__ == "__main__":
    main()
