from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_ORDER = [
    "MLP",
    "GCN",
    "SGC",
    "APPNP",
    "MixHop",
    "GPR-GNN",
    "FAGCN-style",
    "H2GCN",
    "LINKX",
    "HARP-GNN",
    "HARP-X",
    "HARP-SGate",
    "HARP-ESep",
    "HARP-Adaptive",
    "HARP-Blend",
    "HARP-Select",
    "HARP-Projected",
    "HARP-Low",
    "HARP-High",
    "HARP-NoSignal",
    "HARP-BranchGate",
    "HARP-ScalarGate",
    "HARP-ScalarNoSignal",
]
MODEL_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "sgc": "SGC",
    "appnp": "APPNP",
    "mixhop": "MixHop",
    "gprgnn": "GPR-GNN",
    "gpr_gnn": "GPR-GNN",
    "gpr-gnn": "GPR-GNN",
    "fagcn": "FAGCN-style",
    "fagcn_style": "FAGCN-style",
    "fa_gcn": "FAGCN-style",
    "fa-gcn": "FAGCN-style",
    "h2gcn": "H2GCN",
    "linkx": "LINKX",
    "harp": "HARP-GNN",
    "harp_gnn": "HARP-GNN",
    "harp-gnn": "HARP-GNN",
    "harp_x": "HARP-X",
    "harpx": "HARP-X",
    "harp-linkx": "HARP-X",
    "harp_linkx": "HARP-X",
    "harp_sgate": "HARP-SGate",
    "harp_struct_gate": "HARP-SGate",
    "harp-struct-gate": "HARP-SGate",
    "harp_structure_gate": "HARP-SGate",
    "harp_esep": "HARP-ESep",
    "harp_ego_sep": "HARP-ESep",
    "harp-ego-sep": "HARP-ESep",
    "harp_egosep": "HARP-ESep",
    "harp_adaptive": "HARP-Adaptive",
    "harp_ada": "HARP-Adaptive",
    "harp-adaptive": "HARP-Adaptive",
    "harp_adaptive_esep": "HARP-Adaptive",
    "harp_blend": "HARP-Blend",
    "harp_logit_blend": "HARP-Blend",
    "harp-blend": "HARP-Blend",
    "harp_select": "HARP-Select",
    "harp-select": "HARP-Select",
    "harp_projected": "HARP-Projected",
    "harp-projected": "HARP-Projected",
    "harp_fast": "HARP-Projected",
    "harp_low": "HARP-Low",
    "harp_high": "HARP-High",
    "harp_no_signal": "HARP-NoSignal",
    "harp_branch_gate": "HARP-BranchGate",
    "harp_branchgate": "HARP-BranchGate",
    "harp_scalar_gate": "HARP-ScalarGate",
    "harp_scalargate": "HARP-ScalarGate",
    "harp_scalar_no_signal": "HARP-ScalarNoSignal",
    "harp_scalarnosignal": "HARP-ScalarNoSignal",
}


def format_mean_std(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f} $\\pm$ {100.0 * std:.2f}"


def escape_latex_text(value: object) -> str:
    return str(value).replace("_", "\\_")


def summarize(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    grouped = (
        df.groupby(["dataset", "model"])["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["dataset", "mean"], ascending=[True, False])
    )
    grouped["model"] = grouped["model"].map(lambda m: MODEL_LABELS.get(str(m).lower(), str(m)))
    grouped["accuracy"] = grouped.apply(lambda r: format_mean_std(r["mean"], 0.0 if pd.isna(r["std"]) else r["std"]), axis=1)
    best_by_dataset = grouped.groupby("dataset")["mean"].transform("max")
    displayed_tie = (100.0 * grouped["mean"]).round(2) == (100.0 * best_by_dataset).round(2)
    grouped.loc[displayed_tie, "accuracy"] = grouped.loc[displayed_tie, "accuracy"].map(
        lambda value: f"\\textbf{{{value}}}"
    )
    pivot = grouped.pivot(index="dataset", columns="model", values="accuracy")
    ordered_cols = [c for c in MODEL_ORDER if c in pivot.columns]
    ordered_cols.extend(sorted(c for c in pivot.columns if c not in ordered_cols))
    pivot = pivot[ordered_cols]
    pivot.index.name = "Dataset"
    pivot.columns.name = None
    table = pivot.reset_index()
    table["Dataset"] = table["Dataset"].map(escape_latex_text)
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
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment CSV files.")
    parser.add_argument("--input", required=True, help="Input CSV produced by run_experiment.")
    parser.add_argument("--output", required=True, help="Output LaTeX table path.")
    args = parser.parse_args()
    summarize(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
