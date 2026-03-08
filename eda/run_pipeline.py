"""
run_pipeline.py
───────────────
One-command entry point for the full EDA & Feature Engineering pipeline.

Usage
─────
    cd eda/
    python run_pipeline.py          # runs all stages
    python run_pipeline.py --skip-generate   # reuse existing CSVs

Stages executed in order:
  1. generate_data          → raw_sentiment.csv, raw_price.csv
  2. exploratory_analysis   → EDA report + 4 plots
  3. feature_engineering    → processed_dataset.csv, features.csv
"""

import argparse
import sys
import time

from config import RAW_SENTIMENT_CSV, RAW_PRICE_CSV


def banner(text: str) -> None:
    border = "═" * 60
    print(f"\n{border}")
    print(f"  {text}")
    print(f"{border}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full EDA & Feature Engineering pipeline."
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip data generation (reuse existing CSVs).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()

    # ── Stage 1: Data generation ───────────────────────────────────────────
    if args.skip_generate:
        # validate that CSVs exist
        for path, label in [(RAW_SENTIMENT_CSV, "sentiment"), (RAW_PRICE_CSV, "price")]:
            if not path.exists():
                sys.exit(
                    f"[ERROR] --skip-generate was set but {label} CSV is "
                    f"missing at {path}.\n"
                    "       Run without --skip-generate  first."
                )
        print("\n[SKIP] Data generation skipped (--skip-generate)")
    else:
        banner("Stage 1 / 3 — Data Generation")
        from generate_data import main as gen_main
        gen_main()

    # ── Stage 2: Exploratory analysis ──────────────────────────────────────
    banner("Stage 2 / 3 — Exploratory Analysis")
    from exploratory_analysis import main as eda_main
    eda_main()

    # ── Stage 3: Feature engineering ───────────────────────────────────────
    banner("Stage 3 / 3 — Feature Engineering")
    from feature_engineering import main as fe_main
    fe_main()

    # ── Done ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    banner(f"Pipeline finished in {elapsed:.1f}s")
    print()
    print("  Artefacts produced:")
    print("  ───────────────────")
    print("  Data")
    print("    eda/data/raw_sentiment.csv")
    print("    eda/data/raw_price.csv")
    print("  Outputs")
    print("    eda/outputs/processed_dataset.csv")
    print("    eda/outputs/features.csv")
    print("    eda/outputs/eda_report.txt")
    print("  Plots")
    print("    eda/plots/sentiment_histogram.png")
    print("    eda/plots/daily_sentiment_trend.png")
    print("    eda/plots/price_trend.png")
    print("    eda/plots/correlation_plot.png")
    print()


if __name__ == "__main__":
    main()
