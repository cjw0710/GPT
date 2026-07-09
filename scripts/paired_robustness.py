from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from summarize_results import MODEL_LABELS


def _label(model: str) -> str:
    return MODEL_LABELS.get(str(model).lower(), str(model))


def _split_column(df: pd.DataFrame) -> str:
    if "meta_split_id" in df.columns and not df["meta_split_id"].isna().all():
        return "meta_split_id"
    return "seed"


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


def _bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    draws = rng.choice(diff, size=(samples, len(diff)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _wilcoxon_p(diff: np.ndarray) -> float:
    if np.allclose(diff, 0.0):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(
            stats.wilcoxon(
                diff,
                alternative="two-sided",
                zero_method="wilcox",
                method="approx",
            ).pvalue
        )


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


def robust_tests(
    results_path: Path,
    assignments_path: Path,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    results = pd.read_csv(results_path)
    assignments = pd.read_csv(assignments_path)
    key = _split_column(results)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for assignment in assignments.itertuples(index=False):
        dataset = str(assignment.dataset)
        target_model = str(assignment.target_model)
        baseline_model = str(assignment.baseline_model)
        part = results[results["dataset"].astype(str) == dataset]
        target = part[part["model"].astype(str) == target_model][[key, "test_acc"]].rename(
            columns={"test_acc": "target_acc"}
        )
        baseline = part[part["model"].astype(str) == baseline_model][[key, "test_acc"]].rename(
            columns={"test_acc": "baseline_acc"}
        )
        paired = target.merge(baseline, on=key, how="inner").sort_values(key)
        diff = paired["target_acc"].to_numpy(dtype=float) - paired["baseline_acc"].to_numpy(dtype=float)
        if len(diff) < 2:
            raise ValueError(f"Insufficient paired rows for {dataset}: {len(diff)}")
        t_p = 1.0 if np.allclose(diff, 0.0) else float(stats.ttest_rel(
            paired["target_acc"].to_numpy(dtype=float),
            paired["baseline_acc"].to_numpy(dtype=float),
        ).pvalue)
        ci_low, ci_high = _bootstrap_ci(diff, rng, bootstrap_samples)
        rows.append(
            {
                "dataset": dataset,
                "target_model": target_model,
                "baseline_model": baseline_model,
                "n": len(diff),
                "target_mean": float(paired["target_acc"].mean()),
                "baseline_mean": float(paired["baseline_acc"].mean()),
                "diff_mean": float(diff.mean()),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "wins": int((diff > 1e-12).sum()),
                "ties": int((np.abs(diff) <= 1e-12).sum()),
                "losses": int((diff < -1e-12).sum()),
                "t_p_value": t_p,
                "wilcoxon_p_value": _wilcoxon_p(diff),
                "sign_flip_p_value": _exact_sign_flip_p(diff),
            }
        )

    output = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    output["t_p_holm"] = _holm_adjust(output["t_p_value"].to_numpy(dtype=float))
    output["wilcoxon_p_holm"] = _holm_adjust(output["wilcoxon_p_value"].to_numpy(dtype=float))
    output["sign_flip_p_holm"] = _holm_adjust(output["sign_flip_p_value"].to_numpy(dtype=float))
    return output


def _p(value: float) -> str:
    return "$<0.001$" if value < 0.001 else f"{value:.3f}"


def write_latex(results: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{tabular*}{0.94\linewidth}{@{\extracolsep{\fill}}llccccc@{}}",
        r"\toprule",
        r"Dataset & Baseline & Diff (pp) & Bootstrap 95\% CI & W/T/L & Sign-flip $p$ & Holm $p$ \\",
        r"\midrule",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"{row.dataset} & {_label(row.baseline_model)} & "
            f"{100.0 * row.diff_mean:+.2f} & "
            f"[{100.0 * row.bootstrap_ci95_low:+.2f}, {100.0 * row.bootstrap_ci95_high:+.2f}] & "
            f"{row.wins}/{row.ties}/{row.losses} & "
            f"{_p(row.sign_flip_p_value)} & {_p(row.sign_flip_p_holm)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add bootstrap confidence intervals, Wilcoxon tests, and Holm correction to paired results."
    )
    parser.add_argument("--input", required=True, help="Merged per-split result CSV.")
    parser.add_argument("--assignments", required=True, help="CSV selecting the comparison baseline per dataset.")
    parser.add_argument("--output", required=True, help="Output LaTeX table.")
    parser.add_argument("--csv-output", required=True, help="Output robust-statistics CSV.")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    results = robust_tests(
        Path(args.input),
        Path(args.assignments),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_output, index=False)
    write_latex(results, Path(args.output))
    print(results.to_string(index=False))
    print(f"[saved] {csv_output}")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
