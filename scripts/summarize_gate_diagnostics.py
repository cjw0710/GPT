from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    ("diag_gate_mean", "Mean"),
    ("diag_gate_std", "Std."),
    ("diag_gate_q25", "Q25"),
    ("diag_gate_median", "Median"),
    ("diag_gate_q75", "Q75"),
    ("diag_gate_gt_0_5", "Frac. $>0.5$"),
]


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def summarize(input_path: Path, output_path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df[df["model"] == model].copy()
    metric_cols = [column for column, _ in METRICS if column in df.columns]
    if df.empty or not metric_cols:
        raise ValueError(f"No gate diagnostics found for model '{model}' in {input_path}")

    rows: list[dict[str, object]] = []
    for dataset, dataset_df in df.groupby("dataset", sort=True):
        row: dict[str, object] = {"Dataset": dataset}
        for column, label in METRICS:
            if column not in dataset_df.columns:
                continue
            row[label] = format_mean_std(
                float(dataset_df[column].mean()),
                float(dataset_df[column].std(ddof=1)) if len(dataset_df) > 1 else 0.0,
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = table.to_latex(
        index=False,
        escape=False,
        na_rep="--",
        column_format="l" + "c" * (len(table.columns) - 1),
    )
    output_path.write_text(latex, encoding="utf-8")
    print(table)
    print(f"[saved] {output_path}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize HARP gate diagnostics.")
    parser.add_argument("--input", required=True, help="Input CSV produced by run_experiment.")
    parser.add_argument("--output", required=True, help="Output LaTeX table path.")
    parser.add_argument("--csv-output", default=None, help="Optional summary CSV output path.")
    parser.add_argument("--model", default="harp", help="Model name to summarize.")
    args = parser.parse_args()

    table = summarize(Path(args.input), Path(args.output), model=args.model)
    if args.csv_output is not None:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_output, index=False)
        print(f"[saved] {csv_output}")


if __name__ == "__main__":
    main()
