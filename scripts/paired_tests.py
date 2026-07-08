from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from summarize_results import MODEL_LABELS


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model).lower(), str(model))


def split_column(df: pd.DataFrame) -> str:
    if "meta_split_id" in df.columns and not df["meta_split_id"].isna().all():
        return "meta_split_id"
    return "seed"


def paired_p_value(target: np.ndarray, baseline: np.ndarray) -> float:
    diff = target - baseline
    if len(diff) < 2:
        return float("nan")
    if np.allclose(diff, 0.0):
        return 1.0
    if np.isclose(np.std(diff, ddof=1), 0.0):
        return 0.0
    result = stats.ttest_rel(target, baseline, nan_policy="omit")
    return float(result.pvalue)


def format_acc(value: float) -> str:
    return f"{100.0 * value:.2f}"


def format_diff(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def format_p(value: float) -> str:
    if np.isnan(value):
        return "--"
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def best_baseline_tests(
    input_path: Path,
    target_model: str,
    baselines: list[str] | None,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    key = split_column(df)
    rows: list[dict[str, object]] = []

    for dataset, dataset_df in df.groupby("dataset", sort=True):
        available_models = sorted(str(m) for m in dataset_df["model"].unique())
        candidates = baselines if baselines is not None else [m for m in available_models if m != target_model]
        candidates = [m for m in candidates if m in available_models and m != target_model]
        if target_model not in available_models or not candidates:
            continue

        target_df = dataset_df[dataset_df["model"] == target_model][[key, "test_acc"]].rename(
            columns={"test_acc": "target_acc"}
        )
        candidate_rows = []
        for baseline in candidates:
            baseline_df = dataset_df[dataset_df["model"] == baseline][[key, "test_acc"]].rename(
                columns={"test_acc": "baseline_acc"}
            )
            paired = target_df.merge(baseline_df, on=key, how="inner").sort_values(key)
            if paired.empty:
                continue
            target_values = paired["target_acc"].to_numpy(dtype=float)
            baseline_values = paired["baseline_acc"].to_numpy(dtype=float)
            candidate_rows.append(
                {
                    "dataset": dataset,
                    "target_model": target_model,
                    "baseline_model": baseline,
                    "n": len(paired),
                    "target_mean": float(target_values.mean()),
                    "baseline_mean": float(baseline_values.mean()),
                    "diff_mean": float((target_values - baseline_values).mean()),
                    "p_value": paired_p_value(target_values, baseline_values),
                }
            )
        if not candidate_rows:
            continue
        best = max(candidate_rows, key=lambda row: (row["baseline_mean"], model_label(str(row["baseline_model"]))))
        rows.append(best)

    return pd.DataFrame(rows)


def write_latex(results: pd.DataFrame, output_path: Path) -> None:
    target_label = "Target"
    if not results.empty:
        target_label = model_label(str(results["target_model"].iloc[0]))
    display = pd.DataFrame(
        {
            "Dataset": results["dataset"],
            "Best baseline": results["baseline_model"].map(model_label),
            "Baseline": results["baseline_mean"].map(format_acc),
            target_label: results["target_mean"].map(format_acc),
            "Diff (pp)": results["diff_mean"].map(format_diff),
            "$p$": results["p_value"].map(format_p),
        }
    )
    latex = display.to_latex(
        index=False,
        escape=False,
        column_format="llcccc",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    print(display)
    print(f"[saved] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired split-level tests for experiment CSV files.")
    parser.add_argument("--input", required=True, help="Input CSV produced by run_experiment.")
    parser.add_argument("--output", required=True, help="Output LaTeX table path.")
    parser.add_argument("--csv-output", default=None, help="Optional raw CSV output path.")
    parser.add_argument("--target", default="harp", help="Target model name in the CSV.")
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=None,
        help="Optional baseline model names. Defaults to all non-target models.",
    )
    args = parser.parse_args()

    results = best_baseline_tests(Path(args.input), args.target, args.baselines)
    write_latex(results, Path(args.output))
    if args.csv_output is not None:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(csv_output, index=False)
        print(f"[saved] {csv_output}")


if __name__ == "__main__":
    main()
