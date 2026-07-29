"""Aggregate one or more `{paradigm}_explanations.csv` files (as produced by
`explain/run_formal_exp.py`) into per-file mean/std summary statistics for
explanation size and runtime -- the same quantities reported in the paper's Tables 2-4.

Usage:
    python scripts/summarize_results.py path/to/first_explanations.csv path/to/second_explanations.csv ...
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def summarize(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Exp Size"])
    return {
        "file": str(csv_path),
        "n_samples": len(df),
        "accuracy": df["Correct"].mean() if "Correct" in df else float("nan"),
        "exp_size_mean": df["Exp Size"].mean(),
        "exp_size_std": df["Exp Size"].std(),
        "runtime_mean_s": df["Runtime (s)"].mean() if "Runtime (s)" in df else float("nan"),
        "runtime_std_s": df["Runtime (s)"].std() if "Runtime (s)" in df else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", type=Path)
    args = parser.parse_args()

    rows = []
    for csv_path in args.csv_paths:
        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}", file=sys.stderr)
            continue
        rows.append(summarize(csv_path))

    summary = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
