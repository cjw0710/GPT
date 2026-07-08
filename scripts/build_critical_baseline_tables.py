from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from summarize_results import MODEL_LABELS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"

DEFAULT_BASELINES = RESULTS / "critical_heterophily_baselines.csv"
DEFAULT_HARP = RESULTS / "critical_heterophily_harp.csv"
DEFAULT_SELECTOR = RESULTS / "harp_select.csv"
DEFAULT_MERGED = RESULTS / "critical_heterophily_external_with_baselines.csv"
DEFAULT_SUMMARY = RESULTS / "critical_heterophily_external_summary.csv"
DEFAULT_PAIRED = RESULTS / "critical_heterophily_external_best_baseline_tests.csv"
DEFAULT_TABLE = PAPER / "tables" / "critical_heterophily_external_baselines.tex"
DEFAULT_PAIRED_TABLE = PAPER / "tables" / "critical_heterophily_external_best_baseline_tests.tex"
DEFAULT_MAIN_TABLE = PAPER / "tables" / "critical_heterophily_external_main.tex"


TABLE_MODELS = [
    "mlp",
    "gcn",
    "sgc",
    "appnp",
    "mixhop",
    "gprgnn",
    "h2gcn",
    "linkx",
    "harp",
    "harp_esep",
    "harp_select",
]


def _label(model: str) -> str:
    return MODEL_LABELS.get(str(model).lower(), str(model))


def _split_column(df: pd.DataFrame) -> str:
    if "meta_split_id" in df.columns and not df["meta_split_id"].isna().all():
        return "meta_split_id"
    return "seed"


def _read_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in ["seed", "test_acc", "val_acc", "train_acc", "meta_split_id"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["dataset"] = out["dataset"].astype(str)
    out["model"] = out["model"].astype(str)
    return out


def load_external_results(
    baseline_path: Path,
    harp_path: Path,
    selector_path: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in [baseline_path, harp_path, selector_path]:
        frame = _normalize_numeric(_read_if_exists(path))
        if frame.empty:
            continue
        if path == selector_path:
            frame = frame[
                frame["dataset"].astype(str).isin(["roman-empire", "amazon-ratings"])
                & (frame["model"].astype(str) == "harp_select")
            ]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    common_columns = sorted(set.intersection(*(set(frame.columns) for frame in frames)))
    merged = pd.concat([frame[common_columns] for frame in frames], ignore_index=True)
    merged = merged.drop_duplicates(subset=["dataset", "model", "seed"], keep="last")
    return merged.sort_values(["dataset", "model", "seed"]).reset_index(drop=True)


def complete_rows(df: pd.DataFrame, expected_splits: int) -> pd.DataFrame:
    if df.empty:
        return df
    counts = df.groupby(["dataset", "model"])["seed"].nunique().reset_index(name="n")
    complete = counts[counts["n"] >= expected_splits][["dataset", "model"]]
    return df.merge(complete, on=["dataset", "model"], how="inner")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["dataset", "model"])["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["dataset", "mean"], ascending=[True, False])
    )


def _format_mean_std(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f} $\\pm$ {100.0 * (0.0 if pd.isna(std) else std):.2f}"


def write_summary_table(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        raise ValueError("Cannot write table from empty summary.")
    display = summary.copy()
    display["label"] = display["model"].map(_label)
    display["value"] = display.apply(lambda row: _format_mean_std(row["mean"], row["std"]), axis=1)

    best_by_dataset = display.groupby("dataset")["mean"].transform("max")
    is_best = (100.0 * display["mean"]).round(2) == (100.0 * best_by_dataset).round(2)
    display.loc[is_best, "value"] = display.loc[is_best, "value"].map(lambda value: f"\\textbf{{{value}}}")

    pivot = display.pivot(index="dataset", columns="model", values="value")
    ordered = [model for model in TABLE_MODELS if model in pivot.columns]
    ordered.extend(sorted(column for column in pivot.columns if column not in ordered))
    pivot = pivot[ordered]
    pivot = pivot.rename(columns={model: _label(model) for model in pivot.columns})
    table = pivot.reset_index().rename(columns={"dataset": "Dataset"})
    table["Dataset"] = table["Dataset"].map(lambda value: str(value).replace("_", "\\_"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = table.to_latex(index=False, escape=False, na_rep="--", column_format="l" + "c" * (len(table.columns) - 1))
    output_path.write_text(latex, encoding="utf-8")
    print(table.to_string(index=False))
    print(f"[saved] {output_path}")


def _exact_sign_flip_p(diff: np.ndarray) -> float:
    nonzero = diff[np.abs(diff) > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    observed = abs(float(nonzero.sum()))
    total = 2 ** len(nonzero)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        signed = abs(float(np.dot(nonzero, np.asarray(signs, dtype=float))))
        if signed >= observed - 1e-12:
            extreme += 1
    return extreme / total


def _holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def best_baseline_tests(df: pd.DataFrame, target_models: list[str], bootstrap_samples: int, seed: int) -> pd.DataFrame:
    key = _split_column(df)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    summary_df = summarize(df)
    for dataset, dataset_df in df.groupby("dataset", sort=True):
        dataset_summary = summary_df[summary_df["dataset"] == dataset]
        non_harp = dataset_summary[~dataset_summary["model"].astype(str).str.startswith("harp")]
        if non_harp.empty:
            continue
        best_model = str(non_harp.sort_values(["mean", "model"], ascending=[False, True]).iloc[0]["model"])
        baseline = dataset_df[dataset_df["model"] == best_model][[key, "test_acc"]].rename(
            columns={"test_acc": "baseline_acc"}
        )
        for target_model in target_models:
            if target_model not in set(dataset_df["model"].astype(str)):
                continue
            target = dataset_df[dataset_df["model"] == target_model][[key, "test_acc"]].rename(
                columns={"test_acc": "target_acc"}
            )
            paired = target.merge(baseline, on=key, how="inner").sort_values(key)
            if len(paired) < 2:
                continue
            diff = paired["target_acc"].to_numpy(dtype=float) - paired["baseline_acc"].to_numpy(dtype=float)
            draws = rng.choice(diff, size=(bootstrap_samples, len(diff)), replace=True).mean(axis=1)
            t_p = 1.0 if np.allclose(diff, 0.0) else float(
                stats.ttest_rel(
                    paired["target_acc"].to_numpy(dtype=float),
                    paired["baseline_acc"].to_numpy(dtype=float),
                ).pvalue
            )
            rows.append(
                {
                    "dataset": dataset,
                    "target_model": target_model,
                    "baseline_model": best_model,
                    "n": len(paired),
                    "target_mean": float(paired["target_acc"].mean()),
                    "baseline_mean": float(paired["baseline_acc"].mean()),
                    "diff_mean": float(diff.mean()),
                    "bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
                    "wins": int((diff > 1e-12).sum()),
                    "ties": int((np.abs(diff) <= 1e-12).sum()),
                    "losses": int((diff < -1e-12).sum()),
                    "t_p_value": t_p,
                    "sign_flip_p_value": _exact_sign_flip_p(diff),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["t_p_holm"] = _holm(out["t_p_value"].to_numpy(dtype=float))
        out["sign_flip_p_holm"] = _holm(out["sign_flip_p_value"].to_numpy(dtype=float))
    return out


def _p(value: float) -> str:
    if pd.isna(value):
        return "--"
    return "$<0.001$" if value < 0.001 else f"{value:.3f}"


def write_paired_table(tests: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"Dataset & Target & Baseline & Diff (pp) & Bootstrap 95\% CI & W/T/L & Holm $p$ \\",
        r"\midrule",
    ]
    for row in tests.itertuples(index=False):
        lines.append(
            f"{row.dataset} & {_label(row.target_model)} & {_label(row.baseline_model)} & "
            f"{100.0 * row.diff_mean:+.2f} & "
            f"[{100.0 * row.bootstrap_ci95_low:+.2f}, {100.0 * row.bootstrap_ci95_high:+.2f}] & "
            f"{row.wins}/{row.ties}/{row.losses} & {_p(row.sign_flip_p_holm)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {output_path}")


def _summary_lookup(summary: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for row in summary.itertuples(index=False):
        lookup[(str(row.dataset), str(row.model))] = (float(row.mean), float(row.std))
    return lookup


def write_main_external_table(summary: pd.DataFrame, tests: pd.DataFrame, output_path: Path) -> None:
    lookup = _summary_lookup(summary)
    select_tests = tests[tests["target_model"].astype(str) == "harp_select"].copy()
    if select_tests.empty:
        raise ValueError("Cannot build compact external table without HARP-Select paired tests.")

    lines = [
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Dataset & Best non-HARP & Best & HARP-GNN & HARP-ESep & HARP-Select & Diff (pp) & Holm $p$ \\",
        r"\midrule",
    ]
    for row in select_tests.sort_values("dataset").itertuples(index=False):
        dataset = str(row.dataset)
        baseline = str(row.baseline_model)
        best_mean, best_std = lookup[(dataset, baseline)]
        harp_mean, harp_std = lookup[(dataset, "harp")]
        esep_mean, esep_std = lookup[(dataset, "harp_esep")]
        select_mean, select_std = lookup[(dataset, "harp_select")]
        lines.append(
            f"{dataset} & {_label(baseline)} & {_format_mean_std(best_mean, best_std)} & "
            f"{_format_mean_std(harp_mean, harp_std)} & "
            f"{_format_mean_std(esep_mean, esep_std)} & "
            f"{_format_mean_std(select_mean, select_std)} & "
            f"{100.0 * row.diff_mean:+.2f} & {_p(row.sign_flip_p_holm)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external critical-heterophily baseline tables.")
    parser.add_argument("--baselines", default=str(DEFAULT_BASELINES))
    parser.add_argument("--harp", default=str(DEFAULT_HARP))
    parser.add_argument("--selector", default=str(DEFAULT_SELECTOR))
    parser.add_argument("--merged-output", default=str(DEFAULT_MERGED))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--paired-output", default=str(DEFAULT_PAIRED))
    parser.add_argument("--table-output", default=str(DEFAULT_TABLE))
    parser.add_argument("--paired-table-output", default=str(DEFAULT_PAIRED_TABLE))
    parser.add_argument("--main-table-output", default=str(DEFAULT_MAIN_TABLE))
    parser.add_argument("--expected-splits", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()

    merged = load_external_results(Path(args.baselines), Path(args.harp), Path(args.selector))
    complete = complete_rows(merged, expected_splits=args.expected_splits)
    if complete.empty:
        raise SystemExit("[failed] no complete external result groups available")

    merged_output = Path(args.merged_output)
    merged_output.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(merged_output, index=False)
    print(f"[saved] {merged_output}")

    summary_df = summarize(complete)
    summary_output = Path(args.summary_output)
    summary_df.to_csv(summary_output, index=False)
    print(f"[saved] {summary_output}")
    write_summary_table(summary_df, Path(args.table_output))

    tests = best_baseline_tests(
        complete,
        target_models=["harp_esep", "harp_select"],
        bootstrap_samples=args.bootstrap_samples,
        seed=2027,
    )
    paired_output = Path(args.paired_output)
    tests.to_csv(paired_output, index=False)
    print(f"[saved] {paired_output}")
    write_paired_table(tests, Path(args.paired_table_output))
    if not tests.empty:
        write_main_external_table(summary_df, tests, Path(args.main_table_output))
    if tests.empty:
        print("[warn] no paired best-baseline tests were generated")
    else:
        print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
