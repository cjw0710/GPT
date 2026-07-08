from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "SCIENTIFIC_AUDIT.md"

MODEL_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "sgc": "SGC",
    "appnp": "APPNP",
    "mixhop": "MixHop",
    "gprgnn": "GPR-GNN",
    "h2gcn": "H2GCN",
    "linkx": "LINKX",
    "harp": "HARP-GNN",
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
    "harp_low": "HARP-Low",
    "harp_high": "HARP-High",
    "harp_no_signal": "HARP-NoSignal",
    "harp_scalar_gate": "HARP-ScalarGate",
    "harp_scalar_no_signal": "HARP-ScalarNoSignal",
}


@dataclass(frozen=True)
class DatasetSummary:
    dataset: str
    harp_mean: float
    harp_std: float
    best_model: str
    best_mean: float
    best_std: float
    rank: int
    model_count: int


def _label(model: str) -> str:
    return MODEL_LABELS.get(str(model).lower(), str(model))


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _signed_pp(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def _mean_std(mean: float, std: float) -> str:
    return f"{_pct(mean)} +/- {_pct(std)}"


def _p_value(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or f"{singular}s")


def _read_csv(relative: str) -> pd.DataFrame:
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _summaries(df: pd.DataFrame) -> list[DatasetSummary]:
    grouped = (
        df.groupby(["dataset", "model"])["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summaries: list[DatasetSummary] = []
    for dataset, part in grouped.groupby("dataset", sort=True):
        ranked = part.sort_values(["mean", "model"], ascending=[False, True]).reset_index(drop=True)
        harp_rows = ranked[ranked["model"] == "harp"]
        if harp_rows.empty:
            continue
        harp_index = int(harp_rows.index[0])
        harp = ranked.iloc[harp_index]
        best = ranked.iloc[0]
        summaries.append(
            DatasetSummary(
                dataset=str(dataset),
                harp_mean=float(harp["mean"]),
                harp_std=0.0 if pd.isna(harp["std"]) else float(harp["std"]),
                best_model=str(best["model"]),
                best_mean=float(best["mean"]),
                best_std=0.0 if pd.isna(best["std"]) else float(best["std"]),
                rank=harp_index + 1,
                model_count=len(ranked),
            )
        )
    return summaries


def _summary_table(title: str, summaries: list[DatasetSummary]) -> list[str]:
    lines = [f"### {title}", "", "| Dataset | HARP-GNN | Best model | Best acc. | HARP rank | Gap to best (pp) |", "|---|---:|---|---:|---:|---:|"]
    for item in summaries:
        gap = item.harp_mean - item.best_mean
        lines.append(
            "| "
            + " | ".join(
                [
                    item.dataset,
                    _mean_std(item.harp_mean, item.harp_std),
                    _label(item.best_model),
                    _mean_std(item.best_mean, item.best_std),
                    f"{item.rank}/{item.model_count}",
                    _signed_pp(gap),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _paired_section(title: str, df: pd.DataFrame) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Dataset | Baseline | HARP-GNN | Baseline | Diff (pp) | p-value | Interpretation |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in df.sort_values("dataset").itertuples(index=False):
        diff = float(row.diff_mean)
        p_value = float(row.p_value)
        if p_value < 0.05 and diff > 0:
            interpretation = "positive significant"
        elif p_value < 0.05 and diff < 0:
            interpretation = "negative significant"
        elif diff > 0:
            interpretation = "positive, not significant"
        elif diff < 0:
            interpretation = "negative, not significant"
        else:
            interpretation = "tie"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.dataset),
                    _label(str(row.baseline_model)),
                    _pct(float(row.target_mean)),
                    _pct(float(row.baseline_mean)),
                    _signed_pp(diff),
                    f"{p_value:.3g}",
                    interpretation,
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _best_counts(summaries: list[DatasetSummary]) -> tuple[int, int, int]:
    wins = sum(item.rank == 1 for item in summaries)
    close = sum(item.rank != 1 and abs(100.0 * (item.harp_mean - item.best_mean)) <= 1.0 for item in summaries)
    total = len(summaries)
    return wins, close, total


def _ablation_section() -> list[str]:
    ablation = _read_csv("results/webkb_ablation.csv")
    scalar = _read_csv("results/webkb_scalar_gate.csv")
    lines = ["### Ablation Signals", ""]

    def mean_by_model(df: pd.DataFrame, model: str) -> float:
        rows = df[df["model"] == model]
        return float(rows["test_acc"].mean()) if not rows.empty else float("nan")

    harp = mean_by_model(ablation, "harp")
    low = mean_by_model(ablation, "harp_low")
    high = mean_by_model(ablation, "harp_high")
    no_signal = mean_by_model(ablation, "harp_no_signal")
    scalar_gate = mean_by_model(scalar, "harp_scalar_gate")
    scalar_no_signal = mean_by_model(scalar, "harp_scalar_no_signal")

    lines.extend(
        [
            f"- WebKB average HARP-GNN accuracy: {_pct(harp)}.",
            f"- HARP-Low average: {_pct(low)}; HARP-High average: {_pct(high)}. This supports the residual high-pass branch as the main useful component.",
            f"- HARP-NoSignal average: {_pct(no_signal)}. The current handcrafted feature-variation gate signal should remain a bounded claim.",
            f"- Scalar-gate variants average {_pct(scalar_gate)} and {_pct(scalar_no_signal)}, below the feature-wise gate on WebKB.",
            "",
        ]
    )
    return lines


def _efficiency_section() -> list[str]:
    params = _read_csv("results/webkb_parameters.csv").set_index("model")
    runtime = _read_csv("results/webkb_runtime.csv").set_index("model")
    harp_params = float(params.loc["harp", "param_mean"])
    mixhop_params = float(params.loc["mixhop", "param_mean"])
    h2gcn_params = float(params.loc["h2gcn", "param_mean"])
    harp_time = float(runtime.loc["harp", "elapsed_mean"])
    mixhop_time = float(runtime.loc["mixhop", "elapsed_mean"])
    h2gcn_time = float(runtime.loc["h2gcn", "elapsed_mean"])
    lines = [
        "### Efficiency and Capacity",
        "",
        f"- HARP-GNN has {harp_params / 1000.0:.1f}K trainable parameters on WebKB, close to MixHop ({mixhop_params / 1000.0:.1f}K) and larger than H2GCN ({h2gcn_params / 1000.0:.1f}K).",
        f"- CPU time per WebKB split is {harp_time:.2f}s for HARP-GNN, {mixhop_time:.2f}s for MixHop, and {h2gcn_time:.2f}s for H2GCN.",
        "- The paper can claim modest absolute runtime on WebKB, but not speed superiority.",
        "",
    ]
    return lines


def _optional_structure_diagnostic_section() -> list[str]:
    path = ROOT / "results" / "harp_x_diagnostic.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    grouped = (
        df.groupby(["dataset", "model"])["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    lines = [
        "## Structural HARP Diagnostics",
        "",
        "This is a bounded two-seed diagnostic, not a primary manuscript claim.",
        "HARP-X appends a LINKX-style sparse adjacency branch to the residual low/high-pass HARP fusion.",
        "HARP-SGate uses the same sparse adjacency evidence to condition the low/high-pass gate instead of concatenating it into the classifier.",
        "HARP-ESep instead separates ego and no-self-neighbor evidence before applying hidden-space residual HARP filtering.",
        "HARP-Adaptive learns a node-wise selector between the original self-loop HARP branch and the no-self HARP-ESep branch.",
        "HARP-Blend is a more conservative graph-level logit mixture with auxiliary branch supervision.",
        "",
        "| Dataset | HARP-GNN | HARP-X | HARP-SGate | HARP-ESep | HARP-Adaptive | HARP-Blend | Best diagnostic model | Best structural gap (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]

    for dataset, part in grouped.groupby("dataset", sort=True):
        rows = {str(row.model): row for row in part.itertuples(index=False)}
        if "harp" not in rows:
            continue
        harp = rows["harp"]
        best = max(part.itertuples(index=False), key=lambda row: float(row.mean))
        structural_rows = [
            rows[name]
            for name in ("harp_x", "harp_sgate", "harp_esep", "harp_adaptive", "harp_blend")
            if name in rows
        ]
        best_structural = max(structural_rows, key=lambda row: float(row.mean)) if structural_rows else None
        structural_gap = (
            float(best_structural.mean) - float(best.mean)
            if best_structural is not None
            else float("nan")
        )

        def cell(model: str) -> str:
            if model not in rows:
                return "--"
            row = rows[model]
            return _mean_std(float(row.mean), 0.0 if pd.isna(row.std) else float(row.std))

        lines.append(
            "| "
            + " | ".join(
                [
                    str(dataset),
                    _mean_std(float(harp.mean), 0.0 if pd.isna(harp.std) else float(harp.std)),
                    cell("harp_x"),
                    cell("harp_sgate"),
                    cell("harp_esep"),
                    cell("harp_adaptive"),
                    cell("harp_blend"),
                    _label(str(best.model)),
                    "--" if pd.isna(structural_gap) else _signed_pp(structural_gap),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Interpretation: HARP-X and HARP-SGate argue against simple adjacency late fusion or structure-conditioned gates as top-conference fixes.",
            "HARP-ESep is the first promising structural candidate on the larger heterophily rows, and the separate full 10-split run below confirms that signal.",
            "The first two adaptive follow-ups do not close the gap: HARP-Adaptive over-selects ESep on WebKB and under-selects it on Chameleon/Squirrel, while HARP-Blend is stabler on WebKB but too conservative on the larger heterophily rows.",
            "",
        ]
    )
    return lines


def _optional_harp_esep_full_section() -> list[str]:
    result_path = ROOT / "results" / "geom_gcn_harp_esep.csv"
    paired_path = ROOT / "results" / "geom_gcn_harp_esep_paired_tests.csv"
    if not result_path.exists():
        return []

    esep = pd.read_csv(result_path)
    geom = _read_csv("results/geom_gcn_large.csv")
    combined = pd.concat([geom, esep], ignore_index=True)
    grouped = (
        combined.groupby(["dataset", "model"])["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    lines = [
        "## HARP-ESep Full Split Candidate",
        "",
        "HARP-ESep is a 10-split candidate run on Actor, Chameleon, and Squirrel. It is not yet the main manuscript model because it damages WebKB in the two-seed diagnostic, but it directly addresses the larger-heterophily P0 risk.",
        "",
        "| Dataset | HARP-ESep | Best model with candidate | Best acc. | HARP-ESep rank | Gap to best (pp) |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for dataset in ["actor", "chameleon", "squirrel"]:
        part = grouped[grouped["dataset"] == dataset].sort_values(
            ["mean", "model"], ascending=[False, True]
        ).reset_index(drop=True)
        esep_rows = part[part["model"] == "harp_esep"]
        if esep_rows.empty:
            continue
        esep_index = int(esep_rows.index[0])
        row = part.iloc[esep_index]
        best = part.iloc[0]
        gap = float(row["mean"]) - float(best["mean"])
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    _mean_std(float(row["mean"]), 0.0 if pd.isna(row["std"]) else float(row["std"])),
                    _label(str(best["model"])),
                    _mean_std(float(best["mean"]), 0.0 if pd.isna(best["std"]) else float(best["std"])),
                    f"{esep_index + 1}/{len(part)}",
                    _signed_pp(gap),
                ]
            )
            + " |"
        )

    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        lines.extend(
            [
                "",
                "Paired tests against the strongest non-HARP-ESep baseline:",
                "",
                "| Dataset | Baseline | HARP-ESep | Baseline | Diff (pp) | p-value |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in paired.sort_values("dataset").itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.dataset),
                        _label(str(row.baseline_model)),
                        _pct(float(row.target_mean)),
                        _pct(float(row.baseline_mean)),
                        _signed_pp(float(row.diff_mean)),
                        f"{float(row.p_value):.3g}",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "Interpretation: HARP-ESep turns the previous large significant deficits on Chameleon and Squirrel into statistically non-significant gaps against the strongest implemented baselines, and is best on Squirrel in mean accuracy.",
            "The remaining scientific issue is adaptivity: the same no-self ego-separated design hurts WebKB, and the first node-wise/graph-level selectors did not reliably choose the right branch. A publishable main method needs a stronger structural prior or branch-selection calibration.",
            "",
        ]
    )
    return lines


def _optional_harp_select_section() -> list[str]:
    diagnostics_path = ROOT / "results" / "harp_select_diagnostics.csv"
    paired_path = ROOT / "results" / "harp_select_paired_tests.csv"
    robust_path = ROOT / "results" / "harp_select_robust_tests.csv"
    sensitivity_path = ROOT / "results" / "harp_select_threshold_sensitivity_overall.csv"
    calibration_path = ROOT / "results" / "harp_select_margin_calibration.csv"
    cost_path = ROOT / "results" / "harp_select_training_cost.csv"
    external_paired_path = ROOT / "results" / "critical_heterophily_harp_esep_paired_tests.csv"
    external_robust_path = ROOT / "results" / "critical_heterophily_harp_esep_robust_tests.csv"
    if not diagnostics_path.exists() or not paired_path.exists():
        return []

    diagnostics = pd.read_csv(diagnostics_path)
    paired = pd.read_csv(paired_path)
    robust = pd.read_csv(robust_path) if robust_path.exists() else pd.DataFrame()
    sensitivity = pd.read_csv(sensitivity_path) if sensitivity_path.exists() else pd.DataFrame()
    calibration = pd.read_csv(calibration_path) if calibration_path.exists() else pd.DataFrame()
    cost = pd.read_csv(cost_path) if cost_path.exists() else pd.DataFrame()
    external_paired = pd.read_csv(external_paired_path) if external_paired_path.exists() else pd.DataFrame()
    external_robust = pd.read_csv(external_robust_path) if external_robust_path.exists() else pd.DataFrame()
    lines = [
        "## HARP-Select Validation-Calibrated Candidate",
        "",
        "HARP-Select trains self-loop HARP and no-self HARP-ESep independently, then selects HARP-ESep only when its validation advantage exceeds a fixed `1.96 * SE` normal-approximation threshold. Test labels are not used for routing.",
        "",
        "| Dataset | HARP-GNN | HARP-ESep | HARP-Select | ESep splits | Oracle regret (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, part in diagnostics.groupby("dataset", sort=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(dataset),
                    _mean_std(float(part["harp_test_acc"].mean()), float(part["harp_test_acc"].std())),
                    _mean_std(float(part["esep_test_acc"].mean()), float(part["esep_test_acc"].std())),
                    _mean_std(float(part["test_acc"].mean()), float(part["test_acc"].std())),
                    f"{int(part['selected_esep'].sum())}/{len(part)}",
                    f"{100.0 * float(part['oracle_regret'].mean()):.2f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Paired HARP-Select evidence against the strongest implemented non-HARP baseline:",
            "",
            "| Dataset | Baseline | HARP-Select | Baseline | Diff (pp) | p-value |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired.sort_values("dataset").itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.dataset),
                    _label(str(row.baseline_model)),
                    _pct(float(row.target_mean)),
                    _pct(float(row.baseline_mean)),
                    _signed_pp(float(row.diff_mean)),
                    _p_value(float(row.p_value)),
                ]
            )
            + " |"
        )

    significant_robust = 0
    if not robust.empty and "sign_flip_p_holm" in robust.columns:
        significant_robust = int((robust["sign_flip_p_holm"].astype(float) < 0.05).sum())
    lines.extend(
        [
            "",
            f"Robustness readout: exact paired sign-flip tests with Holm correction produce {significant_robust} significant rows at `p < 0.05`.",
            "",
            "Interpretation on the original suite: HARP-Select preserves the original HARP results on Texas, Wisconsin, Cornell, and Actor while routing every Chameleon/Squirrel split to HARP-ESep. It removes the original significant Chameleon/Squirrel deficits, but does not establish statistically significant superiority against the strongest implemented baselines.",
            "",
        ]
    )
    if not sensitivity.empty and not calibration.empty:
        z_rows = sensitivity[sensitivity["z"].round(6) == 1.96]
        high_margin_rows = calibration[calibration["margin_bin"].astype(str) == ">=1.96"]
        if not z_rows.empty and not high_margin_rows.empty:
            z_row = z_rows.iloc[0]
            high_margin = high_margin_rows.iloc[0]
            lines.extend(
                [
                    "Threshold-sensitivity diagnostic:",
                    "",
                    f"- At the frozen manuscript threshold `z=1.96`, the selector chooses HARP-ESep on {int(z_row['esep_selections'])}/80 splits, with macro mean accuracy {100.0 * float(z_row['macro_mean_test_acc']):.2f} and mean oracle regret {100.0 * float(z_row['macro_mean_oracle_regret']):.2f} percentage points.",
                    f"- The high validation-margin bin (`val_diff / SE >= 1.96`) contains {int(high_margin['splits'])} splits and has ESep test-win rate {100.0 * float(high_margin['esep_test_win_rate']):.1f}%.",
                    "- Lower fixed thresholds are reported only as diagnostics; the paper threshold is not retuned after seeing test outcomes.",
                    "",
                ]
            )
    if not cost.empty:
        lines.extend(
            [
                "Training-cost diagnostic:",
                "",
                f"- Recorded CPU wall-clock times give macro mean two-expert training cost {float(cost['two_expert_elapsed_mean_sec'].mean()):.1f} seconds per split and macro overhead {float(cost['overhead_vs_harp'].mean()):.2f}x relative to the self-loop HARP expert.",
                f"- Across all 80 selector runs, the two branches account for {float((cost['two_expert_elapsed_mean_sec'] * cost['splits']).sum()) / 3600.0:.2f} artifact-local wall-clock hours.",
                "- This supports the paper's cost boundary: HARP-Select is an auditable benchmark method, not an efficiency claim.",
                "",
            ]
        )
    if not external_paired.empty:
        lines.extend(
            [
                "External fixed-rule validation on the ICLR 2023 critical-heterophily benchmark:",
                "",
                "| Dataset | HARP-GNN | HARP-ESep | Diff (pp) | p-value | HARP-Select ESep splits |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        external_select_counts = diagnostics.groupby("dataset")["selected_esep"].agg(["sum", "count"])
        for row in external_paired.sort_values("dataset").itertuples(index=False):
            counts = external_select_counts.loc[str(row.dataset)]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.dataset),
                        _pct(float(row.baseline_mean)),
                        _pct(float(row.target_mean)),
                        _signed_pp(float(row.diff_mean)),
                        _p_value(float(row.p_value)),
                        f"{int(counts['sum'])}/{int(counts['count'])}",
                    ]
                )
                + " |"
            )
        roman_holm = float("nan")
        if not external_robust.empty:
            roman_rows = external_robust[external_robust["dataset"].astype(str) == "roman-empire"]
            if not roman_rows.empty:
                roman_holm = float(roman_rows.iloc[0]["sign_flip_p_holm"])
        lines.extend(
            [
                "",
                "External interpretation: Roman-Empire provides the first significant positive branch result: HARP-ESep wins all 10 splits, and the frozen selector routes all 10 splits to it"
                + (
                    f" (Holm-adjusted exact sign-flip p={roman_holm:.4f})."
                    if not pd.isna(roman_holm)
                    else "."
                ),
                "Amazon-Ratings shows the conservative boundary: HARP-ESep has a small non-significant mean advantage, but no split exceeds the fixed confidence threshold, so HARP-Select retains HARP-GNN and incurs modest oracle regret.",
                "The candidate now has external validation, but still lacks strong official-code baselines on Roman-Empire/Amazon-Ratings. The binary ROC-AUC path is implemented for branch comparisons, while selector calibration for ROC-AUC datasets remains future work.",
                "",
            ]
        )
    return lines


def _optional_binary_critical_section() -> list[str]:
    result_path = ROOT / "results" / "critical_heterophily_binary_harp.csv"
    paired_path = ROOT / "results" / "critical_heterophily_binary_complete_paired_tests.csv"
    robust_path = ROOT / "results" / "critical_heterophily_binary_complete_robust_tests.csv"
    smoke_path = ROOT / "results" / "critical_heterophily_binary_smoke.csv"
    if not result_path.exists() or not paired_path.exists() or not robust_path.exists():
        return []

    df = pd.read_csv(result_path)
    paired = pd.read_csv(paired_path).set_index("dataset")
    robust = pd.read_csv(robust_path).set_index("dataset")
    grouped = (
        df.groupby(["dataset", "model"])["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    lines = [
        "## Binary Critical-Heterophily ROC-AUC",
        "",
        "The binary critical-heterophily path evaluates ROC-AUC rather than accuracy. The current manuscript reports complete 10-split Minesweeper and Tolokers branch comparisons; Questions remains smoke-only and is not a main claim.",
        "",
        "| Dataset | Metric | HARP-GNN | HARP-ESep | Diff (pp) | W/T/L | Paired p | Sign-flip Holm p | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in sorted(paired.index.astype(str)):
        part = grouped[grouped["dataset"].astype(str) == dataset]
        rows = {str(row.model): row for row in part.itertuples(index=False)}
        if "harp" not in rows or "harp_esep" not in rows or dataset not in robust.index:
            continue
        paired_row = paired.loc[dataset]
        robust_row = robust.loc[dataset]
        diff = float(paired_row["diff_mean"])
        status = "reported positive branch evidence" if diff > 0 else "reported negative branch evidence"
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    "ROC-AUC",
                    _mean_std(float(rows["harp"].mean), 0.0 if pd.isna(rows["harp"].std) else float(rows["harp"].std)),
                    _mean_std(
                        float(rows["harp_esep"].mean),
                        0.0 if pd.isna(rows["harp_esep"].std) else float(rows["harp_esep"].std),
                    ),
                    _signed_pp(diff),
                    f"{int(robust_row['wins'])}/{int(robust_row['ties'])}/{int(robust_row['losses'])}",
                    _p_value(float(paired_row["p_value"])),
                    _p_value(float(robust_row["sign_flip_p_holm"])),
                    status,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation: Minesweeper supports the ego-separated branch under ROC-AUC, while Tolokers gives equally strong evidence in the opposite direction. These are branch-specialization results, not HARP-Select routing claims. The current selector threshold is derived from validation accuracy uncertainty and should not be reused for ROC-AUC without a separate calibration argument.",
        ]
    )

    if smoke_path.exists():
        smoke = pd.read_csv(smoke_path)
        smoke_datasets = ", ".join(sorted(smoke["dataset"].astype(str).unique()))
        lines.extend(
            [
                f"The smoke file covers {smoke_datasets} with one split per branch and is treated as loader/protocol evidence only.",
            ]
        )
    coverage = df.groupby("dataset")["seed"].nunique().sort_index().to_dict()
    coverage_text = ", ".join(f"{dataset}: {count} seeds" for dataset, count in coverage.items())
    lines.append(f"The full binary branch trace currently contains {len(df)} rows ({coverage_text}).")

    lines.append("")
    return lines


def _top_conference_risk_register(
    wins: int,
    close: int,
    total: int,
    significant_positive: int,
    significant_negative: int,
) -> list[str]:
    status = (
        "The submission-format artifact is internally consistent, but the current "
        "scientific evidence is not yet strong enough for an AAAI main-track claim "
        "of broad competitive superiority."
    )
    return [
        "## Top-Conference Readiness Gate",
        "",
        f"Status: {status}",
        "",
        "Decision distinction:",
        "",
        "- Format/readiness status: green. The paper source, compiled PDF, checklist, packages, and verifiers are synchronized.",
        "- Scientific competitiveness status: red-amber. The current result set is useful and reproducible, but it does not yet clear a top-conference evidence bar.",
        "",
        f"Evidence trigger: HARP-GNN is best on {wins}/{total} {_plural(total, 'row')}, within 1 pp on {close} additional {_plural(close, 'row')}, has {significant_positive} significant positive paired {_plural(significant_positive, 'margin')}, and has {significant_negative} significant negative paired {_plural(significant_negative, 'margin')}.",
        "",
        "| Priority | Risk | Current evidence | Required action before treating the draft as AAAI-main competitive |",
        "|---|---|---|---|",
        "| P0 | Competitive evidence below main-track bar | No significant positive paired margins; significant deficits on Chameleon and Squirrel | Either improve the method or re-scope the contribution to a narrower, explicitly diagnostic claim |",
        "| P0 | Baseline coverage is incomplete for a heterophily paper | Implemented baselines omit official-code FAGCN, BernNet, and newer 2025--2026 heterophily methods | Add license-compatible official or carefully reproduced baselines, with the same fixed splits and audit rows |",
        "| P0 | External strong-baseline coverage is incomplete | Roman-Empire and Amazon-Ratings include the implemented non-HARP suite, but still omit official-code FAGCN, BernNet, and newer 2025--2026 baselines | Add license-compatible official or carefully reproduced newer baselines on the external datasets before making broad competitiveness claims |",
        "| P1 | Binary critical-heterophily selector calibration is incomplete | ROC-AUC support and complete Minesweeper/Tolokers branch comparisons are present, but Questions remains smoke-only and no ROC-AUC-specific selector exists | Finish Questions if claiming the full binary suite, and add a ROC-AUC-specific calibration before applying HARP-Select to binary datasets |",
        "| P1 | Homophily fallback is weak | Synthetic high-homophily and Planetoid checks favor low-pass or simplified propagation baselines | Add an adaptive low-pass fallback, regularizer, or dataset-level branch prior and re-run Planetoid/synthetic checks |",
        "| P1 | Gate signal claim is fragile | HARP-NoSignal is close to the full model and better on Cornell | Treat local feature variation as a hypothesis, or redesign the gate around learned branch representations |",
        "| P1 | Statistical support remains thin | Bootstrap intervals, Wilcoxon tests, exact sign-flip tests, Holm correction, and fixed-threshold sensitivity are available for HARP-Select, but each dataset still has only 10 fixed splits and no significant positive margin | Add external datasets, more repeated seeds where valid, and keep the sensitivity diagnostic synchronized with regenerated selector outputs |",
        "| P2 | Presentation polish can still improve | The framework figure and official PDF are present, but scientific framing is conservative | Keep visual polish, but prioritize stronger evidence over cosmetic changes |",
        "",
        "Go/no-go rule for the next revision:",
        "",
        "- Do not call the draft AAAI-main ready until it either obtains significant positive evidence against strong heterophily baselines on at least several benchmark rows, or reframes itself as a focused diagnostic artifact with a clearly limited claim.",
        "- Keep the manuscript free of state-of-the-art, broad superiority, and significant-gain language unless regenerated results support those claims.",
        "",
    ]


def generate_report(output: Path) -> None:
    synthetic = _summaries(_read_csv("results/synthetic_sweep.csv"))
    planetoid = _summaries(_read_csv("results/planetoid_all.csv"))
    webkb = _summaries(_read_csv("results/webkb.csv"))
    geom = _summaries(_read_csv("results/geom_gcn_large.csv"))
    webkb_paired = _read_csv("results/webkb_paired_tests.csv")
    geom_paired = _read_csv("results/geom_gcn_large_paired_tests.csv")

    all_summaries = synthetic + planetoid + webkb + geom
    wins, close, total = _best_counts(all_summaries)
    significant_negative = int((geom_paired["p_value"] < 0.05).mul(geom_paired["diff_mean"] < 0).sum())
    significant_positive = int((webkb_paired["p_value"] < 0.05).mul(webkb_paired["diff_mean"] > 0).sum())

    lines = [
        "# HARP-GNN Scientific Audit",
        "",
        "This report is generated from the current CSV artifacts. It is meant to keep the paper narrative aligned with the actual evidence before submission.",
        "",
        "## Executive Readout",
        "",
        f"- HARP-GNN is best on {wins}/{total} reported benchmark {_plural(total, 'row')} and within 1 percentage point of the best model on {close} additional {_plural(close, 'row')}.",
        f"- Paired tests currently show {significant_positive} significant positive HARP-GNN {_plural(significant_positive, 'margin')} and {significant_negative} significant negative {_plural(significant_negative, 'margin')}.",
        "- The strongest defensible claim is not state of the art; it is that low/high-pass residual fusion is useful on WebKB-like heterophily and easy to audit.",
        "- The main scientific risk is external validity: Chameleon and Squirrel favor H2GCN/LINKX by significant margins, and Planetoid favors established low-pass or simplified propagation baselines.",
        "",
    ]
    if (ROOT / "results" / "critical_heterophily_binary_minesweeper.csv").exists():
        lines.insert(
            len(lines) - 1,
            "- Binary ROC-AUC support is implemented; Minesweeper and Tolokers have complete 10-split branch comparisons, while Questions remains smoke-only.",
        )
    lines.extend(
        _top_conference_risk_register(
            wins=wins,
            close=close,
            total=total,
            significant_positive=significant_positive,
            significant_negative=significant_negative,
        )
    )
    lines.extend(["## Benchmark Landscape", ""])
    lines.extend(_summary_table("Synthetic Homophily Sweep", synthetic))
    lines.extend(_summary_table("Planetoid Citation Checks", planetoid))
    lines.extend(_summary_table("WebKB Heterophily", webkb))
    lines.extend(_summary_table("Larger Geom-GCN Heterophily", geom))

    lines.extend(["## Paired-Split Evidence", ""])
    lines.extend(_paired_section("WebKB", webkb_paired))
    lines.extend(_paired_section("Actor/Chameleon/Squirrel", geom_paired))

    lines.extend(["## Mechanistic Evidence", ""])
    lines.extend(_ablation_section())
    lines.extend(_efficiency_section())
    lines.extend(_optional_structure_diagnostic_section())
    lines.extend(_optional_harp_esep_full_section())
    lines.extend(_optional_harp_select_section())
    lines.extend(_optional_binary_critical_section())

    lines.extend(
        [
            "## Claim Boundaries for the Current Draft",
            "",
            "Safe claims:",
            "",
            "- HARP-GNN gives a compact residual polynomial formulation with low-pass and high-pass bases.",
            "- On WebKB, HARP-GNN is strongest among the implemented baselines on Texas and Wisconsin, but paired margins are not significant at p < 0.05.",
            "- Learned WebKB filters and gates allocate substantial mass to first-order high-pass residual evidence.",
            "- As a separate candidate diagnostic, HARP-Select uses validation-only confidence routing to remove the original significant Chameleon/Squirrel deficits without creating a significant superiority claim.",
            "- On binary Minesweeper under ROC-AUC, HARP-ESep has a complete 10-split positive branch comparison against HARP-GNN; on Tolokers, the direction reverses and HARP-GNN is better. Neither row extends to a HARP-Select routing claim.",
            "- The artifact includes coverage checks, implementation invariants, manuscript/package checks, and final submission-readiness validation.",
            "",
            "Claims to avoid:",
            "",
            "- Do not claim state-of-the-art performance.",
            "- Do not claim significant WebKB gains.",
            "- Do not claim speed superiority.",
            "- Do not present Planetoid as a tuned citation benchmark result.",
            "- Do not cite Questions as a main binary critical-heterophily result until its full ROC-AUC run is complete.",
            "",
            "## Next Scientific Moves",
            "",
            "1. Add strong official-code baselines on Roman-Empire and Amazon-Ratings under the same fixed masks.",
            "2. Finish Questions under ROC-AUC if claiming the full binary suite, then design a ROC-AUC-specific calibration rule before applying HARP-Select to binary datasets.",
            "3. Investigate distillation, shared encoders, or early branch screening to reduce the measured two-specialist training cost.",
            "4. Improve the low-pass fallback so the model does not lose as much on homophilous/citation graphs.",
            "5. Keep the frozen-threshold sensitivity and calibration diagnostic synchronized, without selecting the threshold on test performance.",
            "",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a scientific audit report from current result CSVs.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output Markdown report path.")
    args = parser.parse_args()
    generate_report(Path(args.output))


if __name__ == "__main__":
    main()
