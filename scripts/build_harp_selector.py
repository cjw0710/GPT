from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.harp_gnn.data import load_dataset  # noqa: E402


PAIR_SPECS = [
    {
        "base_results": ROOT / "results" / "webkb.csv",
        "candidate_results": ROOT / "results" / "webkb_harp_esep.csv",
        "candidate_config": ROOT / "configs" / "webkb_harp_esep.yaml",
    },
    {
        "base_results": ROOT / "results" / "geom_gcn_large.csv",
        "candidate_results": ROOT / "results" / "geom_gcn_harp_esep.csv",
        "candidate_config": ROOT / "configs" / "geom_gcn_harp_esep.yaml",
    },
    {
        "base_results": ROOT / "results" / "critical_heterophily_harp.csv",
        "candidate_results": ROOT / "results" / "critical_heterophily_harp.csv",
        "candidate_config": ROOT / "configs" / "critical_heterophily_harp.yaml",
    },
]


def _name(spec: str | dict[str, Any]) -> str:
    if isinstance(spec, str):
        return spec
    return str(spec["name"])


def _validation_sizes(config_path: Path) -> dict[str, int]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = ROOT / cfg.get("data_root", "data")
    sizes: dict[str, int] = {}
    for spec in cfg["datasets"]:
        dataset = load_dataset(spec, data_root=data_root, seed=0, device=torch.device("cpu"))
        sizes[_name(spec)] = int(dataset.val_idx.numel())
    return sizes


def _branch_rows(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = df[df["model"].astype(str) == model].copy()
    if rows.empty:
        raise ValueError(f"No rows for model={model} in {path}")
    duplicates = rows.duplicated(subset=["dataset", "seed"], keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicate dataset/seed rows for model={model} in {path}")
    return rows


def _format_mean_std(values: pd.Series) -> str:
    return f"{100.0 * values.mean():.2f} $\\pm$ {100.0 * values.std(ddof=1):.2f}"


def _write_summary_table(diagnostics: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{tabular*}{0.94\linewidth}{@{\extracolsep{\fill}}lccccc@{}}",
        r"\toprule",
        r"Dataset & HARP-GNN & HARP-ESep & HARP-Select & Oracle & ESep splits \\",
        r"\midrule",
    ]
    for dataset, part in diagnostics.groupby("dataset", sort=True):
        lines.append(
            f"{dataset} & "
            f"{_format_mean_std(part['harp_test_acc'])} & "
            f"{_format_mean_std(part['esep_test_acc'])} & "
            f"{_format_mean_std(part['test_acc'])} & "
            f"{_format_mean_std(part['oracle_test_acc'])} & "
            f"{int(part['selected_esep'].sum())}/{len(part)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_selector(z_value: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostic_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for pair in PAIR_SPECS:
        base = _branch_rows(pair["base_results"], "harp")
        candidate = _branch_rows(pair["candidate_results"], "harp_esep")
        val_sizes = _validation_sizes(pair["candidate_config"])

        base_map = {
            (str(row["dataset"]), int(row["seed"])): row
            for _, row in base.iterrows()
        }
        candidate_map = {
            (str(row["dataset"]), int(row["seed"])): row
            for _, row in candidate.iterrows()
        }
        if set(base_map) != set(candidate_map):
            missing_base = sorted(set(candidate_map).difference(base_map))
            missing_candidate = sorted(set(base_map).difference(candidate_map))
            raise ValueError(
                f"Branch key mismatch: missing_base={missing_base}, missing_candidate={missing_candidate}"
            )

        for key in sorted(base_map):
            dataset, seed = key
            harp_row = base_map[key]
            esep_row = candidate_map[key]
            val_size = val_sizes[dataset]
            harp_val = float(harp_row["val_acc"])
            esep_val = float(esep_row["val_acc"])
            val_diff = esep_val - harp_val
            standard_error = math.sqrt(
                harp_val * (1.0 - harp_val) / val_size
                + esep_val * (1.0 - esep_val) / val_size
            )
            threshold = z_value * standard_error
            selected_esep = val_diff > threshold
            selected_model = "harp_esep" if selected_esep else "harp"
            source_row = esep_row if selected_esep else harp_row
            harp_test = float(harp_row["test_acc"])
            esep_test = float(esep_row["test_acc"])
            selected_test = float(source_row["test_acc"])
            oracle_test = max(harp_test, esep_test)

            diagnostic_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "val_size": val_size,
                    "selector_z": z_value,
                    "harp_val_acc": harp_val,
                    "esep_val_acc": esep_val,
                    "val_diff": val_diff,
                    "val_diff_standard_error": standard_error,
                    "selection_threshold": threshold,
                    "selected_model": selected_model,
                    "selected_esep": selected_esep,
                    "harp_test_acc": harp_test,
                    "esep_test_acc": esep_test,
                    "test_acc": selected_test,
                    "oracle_test_acc": oracle_test,
                    "oracle_regret": oracle_test - selected_test,
                }
            )

            selected = source_row.to_dict()
            selected["model"] = "harp_select"
            selected["val_acc"] = esep_val if selected_esep else harp_val
            selected["test_acc"] = selected_test
            selected["selector_z"] = z_value
            selected["selector_val_size"] = val_size
            selected["selector_val_diff"] = val_diff
            selected["selector_standard_error"] = standard_error
            selected["selector_threshold"] = threshold
            selected["selector_selected_model"] = selected_model
            selected["selector_oracle_regret"] = oracle_test - selected_test
            selected_rows.append(selected)

    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
    selected = pd.DataFrame(selected_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
    return diagnostics, selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build confidence-calibrated HARP versus HARP-ESep split routing from validation accuracy."
    )
    parser.add_argument("--z", type=float, default=1.96, help="Normal-approximation multiplier.")
    parser.add_argument(
        "--diagnostics-output",
        default="results/harp_select_diagnostics.csv",
        help="Per-split selector diagnostics CSV.",
    )
    parser.add_argument(
        "--results-output",
        default="results/harp_select.csv",
        help="Selected per-split result rows.",
    )
    parser.add_argument(
        "--merged-output",
        default="results/harp_select_with_baselines.csv",
        help="HARP-Select rows merged with implemented benchmark baselines.",
    )
    parser.add_argument(
        "--table-output",
        default="paper/tables/harp_select_results.tex",
        help="LaTeX selector summary table.",
    )
    args = parser.parse_args()

    diagnostics, selected = build_selector(args.z)

    diagnostics_path = ROOT / args.diagnostics_output
    results_path = ROOT / args.results_output
    merged_path = ROOT / args.merged_output
    table_path = ROOT / args.table_output
    for path in (diagnostics_path, results_path, merged_path, table_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    diagnostics.to_csv(diagnostics_path, index=False)
    selected.to_csv(results_path, index=False)
    baselines = pd.concat(
        [
            pd.read_csv(ROOT / "results" / "webkb.csv"),
            pd.read_csv(ROOT / "results" / "geom_gcn_large.csv"),
        ],
        ignore_index=True,
    )
    pd.concat([baselines, selected], ignore_index=True, sort=False).to_csv(merged_path, index=False)
    _write_summary_table(diagnostics, table_path)

    summary = (
        diagnostics.groupby("dataset")
        .agg(
            harp=("harp_test_acc", "mean"),
            esep=("esep_test_acc", "mean"),
            selected=("test_acc", "mean"),
            oracle=("oracle_test_acc", "mean"),
            regret=("oracle_regret", "mean"),
            esep_splits=("selected_esep", "sum"),
            splits=("seed", "count"),
        )
        .reset_index()
    )
    for column in ("harp", "esep", "selected", "oracle", "regret"):
        summary[column] = 100.0 * summary[column]
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"[saved] {diagnostics_path}")
    print(f"[saved] {results_path}")
    print(f"[saved] {merged_path}")
    print(f"[saved] {table_path}")


if __name__ == "__main__":
    main()
