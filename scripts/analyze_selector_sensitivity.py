from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_Z_VALUES = [0.0, 0.5, 1.0, 1.645, 1.96, 2.58]


def _percent(value: float) -> float:
    return 100.0 * float(value)


def _format_pp(value: float) -> str:
    return f"{_percent(value):+.2f}"


def _format_acc(value: float) -> str:
    return f"{_percent(value):.2f}"


def _format_rate(value: float) -> str:
    return f"{_percent(value):.1f}"


def _simulate(diagnostics: pd.DataFrame, z_value: float) -> pd.DataFrame:
    frame = diagnostics.copy()
    threshold = z_value * frame["val_diff_standard_error"].astype(float)
    selected_esep = frame["val_diff"].astype(float) > threshold
    frame["sensitivity_z"] = z_value
    frame["sensitivity_threshold"] = threshold
    frame["sensitivity_selected_esep"] = selected_esep
    frame["sensitivity_test_acc"] = frame["harp_test_acc"].where(
        ~selected_esep, frame["esep_test_acc"]
    )
    frame["sensitivity_oracle_acc"] = frame[["harp_test_acc", "esep_test_acc"]].max(axis=1)
    frame["sensitivity_oracle_regret"] = (
        frame["sensitivity_oracle_acc"] - frame["sensitivity_test_acc"]
    )
    frame["sensitivity_oracle_match"] = (
        frame["sensitivity_oracle_regret"].abs() <= 1e-12
    )
    return frame


def _dataset_summary(simulated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (z_value, dataset), part in simulated.groupby(["sensitivity_z", "dataset"], sort=True):
        rows.append(
            {
                "z": z_value,
                "dataset": dataset,
                "splits": len(part),
                "esep_selections": int(part["sensitivity_selected_esep"].sum()),
                "mean_test_acc": float(part["sensitivity_test_acc"].mean()),
                "std_test_acc": float(part["sensitivity_test_acc"].std(ddof=1)),
                "mean_oracle_regret": float(part["sensitivity_oracle_regret"].mean()),
                "oracle_match_rate": float(part["sensitivity_oracle_match"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _overall_summary(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for z_value, part in dataset_summary.groupby("z", sort=True):
        rows.append(
            {
                "z": z_value,
                "datasets": int(part["dataset"].nunique()),
                "splits": int(part["splits"].sum()),
                "esep_selections": int(part["esep_selections"].sum()),
                "macro_mean_test_acc": float(part["mean_test_acc"].mean()),
                "macro_mean_oracle_regret": float(part["mean_oracle_regret"].mean()),
                "macro_oracle_match_rate": float(part["oracle_match_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _margin_calibration(diagnostics: pd.DataFrame) -> pd.DataFrame:
    frame = diagnostics.copy()
    frame["margin_z"] = frame["val_diff"].astype(float) / frame[
        "val_diff_standard_error"
    ].astype(float)
    frame["test_diff"] = frame["esep_test_acc"].astype(float) - frame["harp_test_acc"].astype(float)
    frame["esep_test_win"] = frame["test_diff"] > 1e-12
    frame["selected_by_1_96"] = frame["margin_z"] > 1.96
    bins = [-math.inf, 0.0, 1.0, 1.96, math.inf]
    labels = ["<0", "[0, 1)", "[1, 1.96)", ">=1.96"]
    frame["margin_bin"] = pd.cut(frame["margin_z"], bins=bins, labels=labels, right=False)

    rows = []
    for label in labels:
        part = frame[frame["margin_bin"].astype(str) == label]
        if part.empty:
            rows.append(
                {
                    "margin_bin": label,
                    "splits": 0,
                    "mean_margin_z": float("nan"),
                    "esep_test_win_rate": float("nan"),
                    "mean_test_diff": float("nan"),
                    "selected_by_1_96_rate": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "margin_bin": label,
                "splits": len(part),
                "mean_margin_z": float(part["margin_z"].mean()),
                "esep_test_win_rate": float(part["esep_test_win"].mean()),
                "mean_test_diff": float(part["test_diff"].mean()),
                "selected_by_1_96_rate": float(part["selected_by_1_96"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _latex_overall_table(overall: pd.DataFrame) -> str:
    rows = [
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "$z$ & ESep splits & Macro acc. & Regret (pp) & Oracle match \\\\",
        "\\midrule",
    ]
    for row in overall.sort_values("z").itertuples(index=False):
        rows.append(
            f"{row.z:g} & {row.esep_selections}/80 & "
            f"{_format_acc(row.macro_mean_test_acc)} & "
            f"{_format_pp(row.macro_mean_oracle_regret)} & "
            f"{_format_rate(row.macro_oracle_match_rate)} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(rows)


def _markdown_report(
    overall: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    calibration: pd.DataFrame,
    z_focus: float,
) -> str:
    lines = [
        "# HARP-Select Threshold Sensitivity",
        "",
        "This diagnostic evaluates fixed selector thresholds on the stored validation and test outcomes.",
        "It is not used to retune the paper threshold. The manuscript rule remains fixed at `z=1.96`.",
        "",
        "## Overall Sensitivity",
        "",
        "| z | ESep selections | Macro test acc. | Mean oracle regret (pp) | Oracle-match rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in overall.sort_values("z").itertuples(index=False):
        lines.append(
            f"| {row.z:g} | {row.esep_selections}/80 | "
            f"{_format_acc(row.macro_mean_test_acc)} | "
            f"{_format_pp(row.macro_mean_oracle_regret)} | "
            f"{_format_rate(row.macro_oracle_match_rate)} |"
        )

    focus = dataset_summary[dataset_summary["z"].round(6) == round(z_focus, 6)].sort_values("dataset")
    lines.extend(
        [
            "",
            f"## Dataset Readout At z={z_focus:g}",
            "",
            "| Dataset | ESep selections | Mean test acc. | Mean oracle regret (pp) | Oracle-match rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in focus.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.esep_selections}/{row.splits} | "
            f"{_format_acc(row.mean_test_acc)} | "
            f"{_format_pp(row.mean_oracle_regret)} | "
            f"{_format_rate(row.oracle_match_rate)} |"
        )

    lines.extend(
        [
            "",
            "## Validation-Margin Calibration",
            "",
            "`margin z` is `val_diff / standard_error`, where positive values favor HARP-ESep.",
            "",
            "| Margin bin | Splits | Mean margin z | ESep test-win rate | Mean ESep-HARP test diff (pp) | Selected by z=1.96 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in calibration.itertuples(index=False):
        if row.splits == 0:
            mean_margin = esep_rate = diff = selected_rate = "--"
        else:
            mean_margin = f"{row.mean_margin_z:.2f}"
            esep_rate = _format_rate(row.esep_test_win_rate)
            diff = _format_pp(row.mean_test_diff)
            selected_rate = _format_rate(row.selected_by_1_96_rate)
        lines.append(
            f"| {row.margin_bin} | {row.splits} | {mean_margin} | "
            f"{esep_rate} | {diff} | {selected_rate} |"
        )

    z196 = overall[overall["z"].round(6) == round(z_focus, 6)].iloc[0]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The fixed manuscript threshold selects HARP-ESep on {int(z196.esep_selections)}/80 splits.",
            "Lower thresholds increase ESep selections and can recover small positive branch advantages, but choosing such a threshold after seeing test outcomes would be post-hoc model selection.",
            "The slightly higher macro average at lower thresholds is therefore diagnostic only; it is not used to change the manuscript rule.",
            "Lower thresholds also risk selecting the no-self expert on regimes where WebKB-style self evidence is important.",
            "The calibration table shows why the paper keeps the conservative rule: large positive validation margins correspond to stable ESep test wins, while small positive margins are mixed and should remain diagnostic rather than a source of test-tuned claims.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    diagnostics_path: Path,
    dataset_output: Path,
    overall_output: Path,
    calibration_output: Path,
    latex_output: Path,
    markdown_output: Path,
    z_values: list[float],
) -> None:
    diagnostics = pd.read_csv(diagnostics_path)
    required = {
        "dataset",
        "val_diff",
        "val_diff_standard_error",
        "harp_test_acc",
        "esep_test_acc",
    }
    missing = required - set(diagnostics.columns)
    if missing:
        raise SystemExit(f"Missing columns in {diagnostics_path}: {sorted(missing)}")

    simulated = pd.concat([_simulate(diagnostics, z) for z in z_values], ignore_index=True)
    dataset_summary = _dataset_summary(simulated)
    overall = _overall_summary(dataset_summary)
    calibration = _margin_calibration(diagnostics)

    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    overall_output.parent.mkdir(parents=True, exist_ok=True)
    calibration_output.parent.mkdir(parents=True, exist_ok=True)
    latex_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)

    dataset_summary.to_csv(dataset_output, index=False)
    overall.to_csv(overall_output, index=False)
    calibration.to_csv(calibration_output, index=False)
    latex_output.write_text(_latex_overall_table(overall), encoding="utf-8")
    markdown_output.write_text(
        _markdown_report(overall, dataset_summary, calibration, z_focus=1.96),
        encoding="utf-8",
    )

    print(f"[saved] {dataset_output}")
    print(f"[saved] {overall_output}")
    print(f"[saved] {calibration_output}")
    print(f"[saved] {latex_output}")
    print(f"[saved] {markdown_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HARP-Select fixed-threshold sensitivity.")
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=ROOT / "results" / "harp_select_diagnostics.csv",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=ROOT / "results" / "harp_select_threshold_sensitivity.csv",
    )
    parser.add_argument(
        "--overall-output",
        type=Path,
        default=ROOT / "results" / "harp_select_threshold_sensitivity_overall.csv",
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=ROOT / "results" / "harp_select_margin_calibration.csv",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=ROOT / "paper" / "tables" / "harp_select_threshold_sensitivity.tex",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "paper" / "HARP_SELECTOR_SENSITIVITY.md",
    )
    parser.add_argument(
        "--z-values",
        default=",".join(str(value) for value in DEFAULT_Z_VALUES),
        help="Comma-separated fixed z multipliers to evaluate.",
    )
    args = parser.parse_args()

    z_values = [float(item.strip()) for item in args.z_values.split(",") if item.strip()]
    if 1.96 not in {round(value, 6) for value in z_values}:
        z_values.append(1.96)
    run(
        diagnostics_path=args.diagnostics,
        dataset_output=args.dataset_output,
        overall_output=args.overall_output,
        calibration_output=args.calibration_output,
        latex_output=args.latex_output,
        markdown_output=args.markdown_output,
        z_values=sorted(set(z_values)),
    )


if __name__ == "__main__":
    main()
