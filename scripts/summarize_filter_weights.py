from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

from summarize_results import MODEL_LABELS


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model).lower(), str(model))


def parse_weights(value: object) -> dict[str, list[float]] | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict) or "low" not in parsed or "high" not in parsed:
        return None
    return {
        "low": [float(v) for v in parsed["low"]],
        "high": [float(v) for v in parsed["high"]],
    }


def tidy_weights(input_path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for _, row in df[df["model"] == model].iterrows():
        weights = parse_weights(row.get("filter_weights"))
        if weights is None:
            continue
        for branch, values in weights.items():
            for idx, value in enumerate(values):
                order = idx if branch == "low" else idx + 1
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "model": row["model"],
                        "branch": branch,
                        "order": order,
                        "weight": value,
                    }
                )
    return pd.DataFrame(rows)


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def summarize(input_path: Path, output_path: Path, model: str) -> pd.DataFrame:
    tidy = tidy_weights(input_path, model=model)
    if tidy.empty:
        raise ValueError(f"No filter weights found for model '{model}' in {input_path}")

    grouped = (
        tidy.groupby(["dataset", "branch", "order"])["weight"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["basis"] = grouped.apply(
        lambda row: f"L{int(row['order'])}" if row["branch"] == "low" else f"H{int(row['order'])}",
        axis=1,
    )
    grouped["weight"] = grouped.apply(
        lambda row: format_mean_std(float(row["mean"]), 0.0 if pd.isna(row["std"]) else float(row["std"])),
        axis=1,
    )

    pivot = grouped.pivot(index="dataset", columns="basis", values="weight")
    low_cols = sorted((c for c in pivot.columns if c.startswith("L")), key=lambda c: int(c[1:]))
    high_cols = sorted((c for c in pivot.columns if c.startswith("H")), key=lambda c: int(c[1:]))
    pivot = pivot[low_cols + high_cols]
    pivot.index.name = "Dataset"
    pivot.columns.name = None
    table = pivot.reset_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = table.to_latex(
        index=False,
        escape=False,
        na_rep="--",
        column_format=r"@{\extracolsep{\fill}}" + "l" + "c" * (len(table.columns) - 1) + "@{}",
    )
    latex = latex.replace(r"\begin{tabular}", r"\begin{tabular*}{0.94\linewidth}", 1)
    latex = latex.replace(r"\end{tabular}", r"\end{tabular*}", 1)
    output_path.write_text(latex, encoding="utf-8")
    print(f"Model: {model_label(model)}")
    print(table)
    print(f"[saved] {output_path}")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize learned HARP filter weights.")
    parser.add_argument("--input", required=True, help="Input CSV produced by run_experiment.")
    parser.add_argument("--output", required=True, help="Output LaTeX table path.")
    parser.add_argument("--csv-output", default=None, help="Optional tidy CSV output path.")
    parser.add_argument("--model", default="harp", help="Model name to summarize.")
    args = parser.parse_args()

    grouped = summarize(Path(args.input), Path(args.output), model=args.model)
    if args.csv_output is not None:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(csv_output, index=False)
        print(f"[saved] {csv_output}")


if __name__ == "__main__":
    main()
