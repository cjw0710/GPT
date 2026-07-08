from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PAIR_SPECS = [
    (
        ROOT / "results" / "webkb.csv",
        ROOT / "results" / "webkb_harp_esep.csv",
    ),
    (
        ROOT / "results" / "geom_gcn_large.csv",
        ROOT / "results" / "geom_gcn_harp_esep.csv",
    ),
    (
        ROOT / "results" / "critical_heterophily_harp.csv",
        ROOT / "results" / "critical_heterophily_harp.csv",
    ),
]


def _branch_rows(path: Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows = frame[frame["model"].astype(str) == model].copy()
    if rows.empty:
        raise SystemExit(f"No rows for model={model} in {path}")
    required = {"dataset", "seed", "elapsed_sec"}
    missing = required - set(rows.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns for cost analysis: {sorted(missing)}")
    duplicates = rows.duplicated(subset=["dataset", "seed"], keep=False)
    if duplicates.any():
        raise SystemExit(f"Duplicate dataset/seed rows for model={model} in {path}")
    return rows[["dataset", "seed", "elapsed_sec"]].rename(
        columns={"elapsed_sec": f"{model}_elapsed_sec"}
    )


def _format_sec(value: float) -> str:
    return f"{float(value):.1f}"


def _format_factor(value: float) -> str:
    return f"{float(value):.2f}$\\times$"


def _build_cost_frame(diagnostics: pd.DataFrame) -> pd.DataFrame:
    merged_parts = []
    for base_path, esep_path in PAIR_SPECS:
        harp = _branch_rows(base_path, "harp")
        esep = _branch_rows(esep_path, "harp_esep")
        merged = harp.merge(esep, on=["dataset", "seed"], how="inner", validate="one_to_one")
        if len(merged) != len(harp) or len(merged) != len(esep):
            raise SystemExit(f"Cost branch key mismatch for {base_path} and {esep_path}")
        merged_parts.append(merged)

    per_split = pd.concat(merged_parts, ignore_index=True)
    decisions = diagnostics[["dataset", "seed", "selected_model", "selected_esep"]].copy()
    per_split = per_split.merge(decisions, on=["dataset", "seed"], how="inner", validate="one_to_one")
    if len(per_split) != len(decisions):
        raise SystemExit("Cost rows do not cover all HARP-Select decisions")
    per_split["two_expert_elapsed_sec"] = (
        per_split["harp_elapsed_sec"] + per_split["harp_esep_elapsed_sec"]
    )
    per_split["selected_branch_elapsed_sec"] = per_split["harp_elapsed_sec"].where(
        ~per_split["selected_esep"].astype(bool), per_split["harp_esep_elapsed_sec"]
    )
    per_split["overhead_vs_harp"] = (
        per_split["two_expert_elapsed_sec"] / per_split["harp_elapsed_sec"]
    )
    per_split["esep_cost_share"] = (
        per_split["harp_esep_elapsed_sec"] / per_split["two_expert_elapsed_sec"]
    )
    return per_split.sort_values(["dataset", "seed"]).reset_index(drop=True)


def _summary(per_split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, part in per_split.groupby("dataset", sort=True):
        rows.append(
            {
                "dataset": dataset,
                "splits": len(part),
                "harp_elapsed_mean_sec": float(part["harp_elapsed_sec"].mean()),
                "esep_elapsed_mean_sec": float(part["harp_esep_elapsed_sec"].mean()),
                "two_expert_elapsed_mean_sec": float(part["two_expert_elapsed_sec"].mean()),
                "selected_branch_elapsed_mean_sec": float(part["selected_branch_elapsed_sec"].mean()),
                "overhead_vs_harp": float(part["overhead_vs_harp"].mean()),
                "esep_cost_share": float(part["esep_cost_share"].mean()),
                "esep_selections": int(part["selected_esep"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _latex_table(summary: pd.DataFrame) -> str:
    rows = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Dataset & HARP sec. & ESep sec. & Two-expert sec. & Cost / HARP \\\\",
        "\\midrule",
    ]
    for row in summary.sort_values("dataset").itertuples(index=False):
        rows.append(
            f"{row.dataset} & "
            f"{_format_sec(row.harp_elapsed_mean_sec)} & "
            f"{_format_sec(row.esep_elapsed_mean_sec)} & "
            f"{_format_sec(row.two_expert_elapsed_mean_sec)} & "
            f"{_format_factor(row.overhead_vs_harp)} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(rows)


def _markdown_report(summary: pd.DataFrame, per_split: pd.DataFrame) -> str:
    macro = {
        "harp": float(summary["harp_elapsed_mean_sec"].mean()),
        "esep": float(summary["esep_elapsed_mean_sec"].mean()),
        "two": float(summary["two_expert_elapsed_mean_sec"].mean()),
        "overhead": float(summary["overhead_vs_harp"].mean()),
    }
    total = {
        "harp": float(per_split["harp_elapsed_sec"].sum()),
        "esep": float(per_split["harp_esep_elapsed_sec"].sum()),
        "two": float(per_split["two_expert_elapsed_sec"].sum()),
    }
    lines = [
        "# HARP-Select Training Cost Diagnostic",
        "",
        "This diagnostic reads the recorded `elapsed_sec` values from the reported branch runs.",
        "It does not benchmark new hardware runs and should be interpreted as artifact-local CPU wall-clock evidence.",
        "",
        "## Dataset-Level Cost",
        "",
        "| Dataset | HARP sec./split | ESep sec./split | Two-expert sec./split | Cost / HARP | ESep selections |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("dataset").itertuples(index=False):
        lines.append(
            f"| {row.dataset} | "
            f"{row.harp_elapsed_mean_sec:.1f} | "
            f"{row.esep_elapsed_mean_sec:.1f} | "
            f"{row.two_expert_elapsed_mean_sec:.1f} | "
            f"{row.overhead_vs_harp:.2f}x | "
            f"{row.esep_selections}/{row.splits} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Readout",
            "",
            f"- Macro mean HARP time: {macro['harp']:.1f} seconds per split.",
            f"- Macro mean ESep time: {macro['esep']:.1f} seconds per split.",
            f"- Macro mean two-expert HARP-Select training time: {macro['two']:.1f} seconds per split.",
            f"- Macro mean overhead versus self-loop HARP: {macro['overhead']:.2f}x.",
            f"- Across all 80 selector runs, the artifact records {total['two'] / 3600.0:.2f} wall-clock hours for the two experts combined.",
            "",
            "## Interpretation",
            "",
            "HARP-Select is an auditable benchmark method, not an efficiency claim.",
            "Its selection rule requires both branches to be trained before the validation decision, so deployment-oriented versions should use shared encoders, early branch screening, or distillation.",
            "The cost is still modest enough for the reported fixed-split study: WebKB rows are lightweight, while the external datasets and Squirrel dominate the wall-clock total.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    diagnostics_path: Path,
    per_split_output: Path,
    summary_output: Path,
    latex_output: Path,
    markdown_output: Path,
) -> None:
    diagnostics = pd.read_csv(diagnostics_path)
    per_split = _build_cost_frame(diagnostics)
    summary = _summary(per_split)

    for path in (per_split_output, summary_output, latex_output, markdown_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    per_split.to_csv(per_split_output, index=False)
    summary.to_csv(summary_output, index=False)
    latex_output.write_text(_latex_table(summary), encoding="utf-8")
    markdown_output.write_text(_markdown_report(summary, per_split), encoding="utf-8")

    print(f"[saved] {per_split_output}")
    print(f"[saved] {summary_output}")
    print(f"[saved] {latex_output}")
    print(f"[saved] {markdown_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze recorded HARP-Select two-expert training cost.")
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=ROOT / "results" / "harp_select_diagnostics.csv",
    )
    parser.add_argument(
        "--per-split-output",
        type=Path,
        default=ROOT / "results" / "harp_select_training_cost_per_split.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "results" / "harp_select_training_cost.csv",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=ROOT / "paper" / "tables" / "harp_select_training_cost.tex",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "paper" / "HARP_SELECTOR_COST.md",
    )
    args = parser.parse_args()

    run(
        diagnostics_path=args.diagnostics,
        per_split_output=args.per_split_output,
        summary_output=args.summary_output,
        latex_output=args.latex_output,
        markdown_output=args.markdown_output,
    )


if __name__ == "__main__":
    main()
