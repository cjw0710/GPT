from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from summarize_results import MODEL_LABELS


ROOT = Path(__file__).resolve().parents[1]


def _label(model: str) -> str:
    return MODEL_LABELS.get(str(model).lower(), str(model))


def _split_column(df: pd.DataFrame) -> str:
    if "meta_split_id" in df.columns and not df["meta_split_id"].isna().all():
        return "meta_split_id"
    return "seed"


def _metric_column(df: pd.DataFrame) -> str:
    if "test_roc_auc" in df.columns and not df["test_roc_auc"].isna().all():
        return "test_roc_auc"
    if "test_metric" in df.columns and not df["test_metric"].isna().all():
        return "test_metric"
    return "test_acc"


def _paired_p(target: np.ndarray, baseline: np.ndarray) -> float:
    diff = target - baseline
    if len(diff) < 2:
        return float("nan")
    if np.allclose(diff, 0.0):
        return 1.0
    if np.isclose(np.std(diff, ddof=1), 0.0):
        return 0.0
    return float(stats.ttest_rel(target, baseline, nan_policy="omit").pvalue)


def _exact_sign_flip_p(diff: np.ndarray) -> float:
    nonzero = diff[np.abs(diff) > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    observed = abs(float(nonzero.sum()))
    extreme = 0
    total = 2 ** len(nonzero)
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        signed_sum = abs(float(np.dot(nonzero, np.asarray(signs, dtype=float))))
        if signed_sum >= observed - 1e-12:
            extreme += 1
    return extreme / total


def _bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    draws = rng.choice(diff, size=(samples, len(diff)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _p(value: float) -> str:
    if np.isnan(value):
        return "--"
    return "$<0.001$" if value < 0.001 else f"{value:.3f}"


def _format_mean_std(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f} $\\pm$ {100.0 * std:.2f}"


def _dataset_rows(
    df: pd.DataFrame,
    dataset: str,
    target_model: str,
    baseline_model: str,
    expected_splits: int,
) -> tuple[dict[str, object] | None, np.ndarray | None, str]:
    split_key = _split_column(df)
    metric_key = _metric_column(df)
    part = df[df["dataset"].astype(str) == dataset]
    target = part[part["model"].astype(str) == target_model][[split_key, metric_key]].rename(
        columns={metric_key: "target_metric"}
    )
    baseline = part[part["model"].astype(str) == baseline_model][[split_key, metric_key]].rename(
        columns={metric_key: "baseline_metric"}
    )
    paired = target.merge(baseline, on=split_key, how="inner").sort_values(split_key)
    if len(paired) != expected_splits:
        return None, None, f"{dataset}: skipped, paired splits={len(paired)}/{expected_splits}"

    target_values = paired["target_metric"].to_numpy(dtype=float)
    baseline_values = paired["baseline_metric"].to_numpy(dtype=float)
    diff = target_values - baseline_values
    row = {
        "dataset": dataset,
        "target_model": target_model,
        "baseline_model": baseline_model,
        "n": len(diff),
        "target_mean": float(target_values.mean()),
        "target_std": float(target_values.std(ddof=1)),
        "baseline_mean": float(baseline_values.mean()),
        "baseline_std": float(baseline_values.std(ddof=1)),
        "diff_mean": float(diff.mean()),
        "p_value": _paired_p(target_values, baseline_values),
    }
    return row, diff, f"{dataset}: included, paired splits={len(paired)}"


def build_tables(
    input_path: Path,
    paired_tex: Path,
    paired_csv: Path,
    robust_tex: Path,
    robust_csv: Path,
    target_model: str,
    baseline_model: str,
    expected_splits: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(input_path)
    if "metric_name" in df.columns:
        metrics = {str(value).lower() for value in df["metric_name"].dropna().unique()}
        if metrics and metrics != {"roc_auc"}:
            raise ValueError(f"Expected only ROC-AUC rows, got metric_name={sorted(metrics)}")

    rng = np.random.default_rng(seed)
    paired_rows: list[dict[str, object]] = []
    robust_rows: list[dict[str, object]] = []
    messages: list[str] = []
    datasets = sorted(str(dataset) for dataset in df["dataset"].dropna().unique())
    for dataset in datasets:
        row, diff, message = _dataset_rows(df, dataset, target_model, baseline_model, expected_splits)
        messages.append(message)
        if row is None or diff is None:
            continue
        paired_rows.append(row)
        ci_low, ci_high = _bootstrap_ci(diff, rng, bootstrap_samples)
        robust_rows.append(
            {
                **row,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "wins": int((diff > 1e-12).sum()),
                "ties": int((np.abs(diff) <= 1e-12).sum()),
                "losses": int((diff < -1e-12).sum()),
                "sign_flip_p_value": _exact_sign_flip_p(diff),
            }
        )

    if not paired_rows:
        raise ValueError("No complete binary critical-heterophily datasets were found.")

    paired = pd.DataFrame(paired_rows).sort_values("dataset").reset_index(drop=True)
    robust = pd.DataFrame(robust_rows).sort_values("dataset").reset_index(drop=True)
    robust["sign_flip_p_holm"] = _holm_adjust(robust["sign_flip_p_value"].to_numpy(dtype=float))

    paired_csv.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(paired_csv, index=False)
    robust_csv.parent.mkdir(parents=True, exist_ok=True)
    robust.to_csv(robust_csv, index=False)
    _write_paired_latex(paired, paired_tex)
    _write_robust_latex(robust, robust_tex)
    return paired, robust, messages


def _write_paired_latex(results: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccc@{}}",
        r"\toprule",
        r"Dataset & HARP-GNN & HARP-ESep & Diff (pp) & $n$ & $p$ \\",
        r"\midrule",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"{row.dataset} & "
            f"{_format_mean_std(row.baseline_mean, row.baseline_std)} & "
            f"{_format_mean_std(row.target_mean, row.target_std)} & "
            f"{100.0 * row.diff_mean:+.2f} & "
            f"{row.n} & {_p(row.p_value)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_robust_latex(results: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Dataset & Diff (pp) & Bootstrap 95\% CI & W/T/L & Sign-flip Holm $p$ \\",
        r"\midrule",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"{row.dataset} & "
            f"{100.0 * row.diff_mean:+.2f} & "
            f"[{100.0 * row.bootstrap_ci95_low:+.2f}, {100.0 * row.bootstrap_ci95_high:+.2f}] & "
            f"{row.wins}/{row.ties}/{row.losses} & "
            f"{_p(row.sign_flip_p_holm)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build complete-only ROC-AUC tables for binary critical-heterophily branch comparisons."
    )
    parser.add_argument("--input", default="results/critical_heterophily_binary_harp.csv")
    parser.add_argument("--paired-tex", default="paper/tables/critical_heterophily_binary_complete_paired_tests.tex")
    parser.add_argument("--paired-csv", default="results/critical_heterophily_binary_complete_paired_tests.csv")
    parser.add_argument("--robust-tex", default="paper/tables/critical_heterophily_binary_complete_robust_tests.tex")
    parser.add_argument("--robust-csv", default="results/critical_heterophily_binary_complete_robust_tests.csv")
    parser.add_argument("--target", default="harp_esep")
    parser.add_argument("--baseline", default="harp")
    parser.add_argument("--expected-splits", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    paired, robust, messages = build_tables(
        input_path=ROOT / args.input,
        paired_tex=ROOT / args.paired_tex,
        paired_csv=ROOT / args.paired_csv,
        robust_tex=ROOT / args.robust_tex,
        robust_csv=ROOT / args.robust_csv,
        target_model=args.target,
        baseline_model=args.baseline,
        expected_splits=args.expected_splits,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    for message in messages:
        print(message)
    print(paired.to_string(index=False))
    print(robust[["dataset", "diff_mean", "wins", "ties", "losses", "sign_flip_p_holm"]].to_string(index=False))
    print("[saved] complete-only binary critical-heterophily tables")


if __name__ == "__main__":
    main()
