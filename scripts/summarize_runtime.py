from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from summarize_results import MODEL_LABELS, MODEL_ORDER


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model).lower(), str(model))


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def summarize(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required = {"model", "elapsed_sec", "best_epoch"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing)}")

    grouped = (
        df.groupby("model")
        .agg(
            elapsed_mean=("elapsed_sec", "mean"),
            elapsed_std=("elapsed_sec", "std"),
            epoch_mean=("best_epoch", "mean"),
            runs=("elapsed_sec", "count"),
        )
        .reset_index()
    )
    grouped["Model"] = grouped["model"].map(model_label)
    grouped["Time / split (s)"] = grouped.apply(
        lambda row: format_mean_std(
            float(row["elapsed_mean"]),
            0.0 if pd.isna(row["elapsed_std"]) else float(row["elapsed_std"]),
        ),
        axis=1,
    )
    grouped["Best epoch"] = grouped["epoch_mean"].map(lambda value: f"{float(value):.1f}")
    grouped["Runs"] = grouped["runs"].astype(int)
    order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    grouped["order"] = grouped["Model"].map(lambda value: order.get(value, len(order)))
    grouped = grouped.sort_values(["order", "Model"])
    table = grouped[["Model", "Time / split (s)", "Best epoch", "Runs"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = table.to_latex(
        index=False,
        escape=False,
        column_format="lccc",
    )
    output_path.write_text(latex, encoding="utf-8")
    print(table)
    print(f"[saved] {output_path}")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-run training runtime.")
    parser.add_argument("--input", required=True, help="Input CSV produced by run_experiment.")
    parser.add_argument("--output", required=True, help="Output LaTeX table path.")
    parser.add_argument("--csv-output", default=None, help="Optional raw CSV output path.")
    args = parser.parse_args()

    grouped = summarize(Path(args.input), Path(args.output))
    if args.csv_output is not None:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(csv_output, index=False)
        print(f"[saved] {csv_output}")


if __name__ == "__main__":
    main()
