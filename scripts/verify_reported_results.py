from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _name(spec: str | dict[str, Any]) -> str:
    if isinstance(spec, str):
        return spec
    return str(spec["name"])


def _project_root(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.parent.name == "configs":
        return config_path.parents[1]
    return config_path.parent


def _read_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _expected_rows(config_path: Path) -> tuple[set[tuple[str, str, int]], Path]:
    cfg = _read_config(config_path)
    datasets = [_name(spec) for spec in cfg["datasets"]]
    models = [_name(spec) for spec in cfg["models"]]
    seeds = [int(seed) for seed in cfg.get("seeds", [0])]
    output = _project_root(config_path) / cfg.get("output", "results/results.csv")
    expected = {(dataset, model, seed) for dataset in datasets for model in models for seed in seeds}
    return expected, output


def _read_csv(path: str) -> pd.DataFrame:
    csv_path = ROOT / path
    if not csv_path.exists():
        raise AssertionError(f"Missing CSV: {csv_path}")
    return pd.read_csv(csv_path)


def _assert_close(actual: float, expected: float, label: str, tol: float = 0.005) -> None:
    if not math.isclose(actual, expected, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected:.6f}, got {actual:.6f}")


def _assert_contains(path: str, needle: str) -> None:
    text_path = ROOT / path
    if not text_path.exists():
        raise AssertionError(f"Missing text file: {text_path}")
    text = text_path.read_text(encoding="utf-8")
    if needle not in text:
        raise AssertionError(f"{path} does not contain expected text: {needle}")


def _metric_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["dataset", "model"])["test_acc"].agg(["mean", "std", "count"])


def check_config_coverage(config_path: str) -> list[str]:
    path = ROOT / config_path
    expected, results_path = _expected_rows(path)
    if not results_path.exists():
        raise AssertionError(f"Missing results file for {config_path}: {results_path}")
    df = pd.read_csv(results_path)
    observed = {
        (str(row.dataset), str(row.model), int(row.seed))
        for row in df[["dataset", "model", "seed"]].itertuples(index=False)
    }
    duplicate_count = int(df.duplicated(subset=["dataset", "model", "seed"], keep=False).sum())
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if missing or extra or duplicate_count:
        raise AssertionError(
            f"{config_path}: missing={len(missing)} extra={len(extra)} duplicates={duplicate_count}"
        )
    return [f"{config_path}: {len(expected)} expected rows covered"]


def check_result_audit() -> list[str]:
    df = _read_csv("results/result_audit.csv")
    bad = df[df["status"] != "complete"]
    if not bad.empty:
        raise AssertionError("Non-complete configs in result_audit.csv:\n" + bad.to_string(index=False))
    return [f"result_audit.csv: all {len(df)} configs complete"]


def check_geom_gcn_large() -> list[str]:
    df = _read_csv("results/geom_gcn_large.csv")
    if len(df) != 270:
        raise AssertionError(f"geom_gcn_large.csv: expected 270 rows, got {len(df)}")
    metrics = _metric_table(df)
    expected = {
        ("actor", "linkx"): (35.67, 1.22, 10),
        ("actor", "harp"): (35.08, 1.36, 10),
        ("chameleon", "h2gcn"): (64.21, 2.66, 10),
        ("chameleon", "harp"): (55.90, 2.49, 10),
        ("squirrel", "linkx"): (45.48, 2.33, 10),
        ("squirrel", "h2gcn"): (45.12, 1.71, 10),
        ("squirrel", "harp"): (36.66, 1.66, 10),
    }
    for key, (mean, std, count) in expected.items():
        actual = metrics.loc[key]
        _assert_close(float(actual["mean"]) * 100.0, mean, f"{key} mean")
        _assert_close(float(actual["std"]) * 100.0, std, f"{key} std")
        if int(actual["count"]) != count:
            raise AssertionError(f"{key} count: expected {count}, got {int(actual['count'])}")

    paired = _read_csv("results/geom_gcn_large_paired_tests.csv").set_index("dataset")
    paired_expected: dict[str, tuple[str, float, float | str]] = {
        "actor": ("linkx", -0.59, 0.245),
        "chameleon": ("h2gcn", -8.31, "<0.001"),
        "squirrel": ("linkx", -8.82, "<0.001"),
    }
    for dataset, (baseline, diff_pp, p_value) in paired_expected.items():
        row = paired.loc[dataset]
        if str(row["baseline_model"]) != baseline:
            raise AssertionError(f"{dataset} paired baseline: expected {baseline}, got {row['baseline_model']}")
        _assert_close(float(row["diff_mean"]) * 100.0, diff_pp, f"{dataset} paired diff")
        actual_p = float(row["p_value"])
        if p_value == "<0.001":
            if not actual_p < 0.001:
                raise AssertionError(f"{dataset} paired p: expected <0.001, got {actual_p:.6f}")
        else:
            _assert_close(actual_p, float(p_value), f"{dataset} paired p", tol=0.0005)

    table_checks = [
        "35.67 $\\pm$ 1.22",
        "35.08 $\\pm$ 1.36",
        "64.21 $\\pm$ 2.66",
        "55.90 $\\pm$ 2.49",
        "45.48 $\\pm$ 2.33",
        "36.66 $\\pm$ 1.66",
    ]
    for needle in table_checks:
        _assert_contains("paper/tables/geom_gcn_large_results.tex", needle)
    _assert_contains("paper/tables/geom_gcn_large_paired_tests.tex", "actor & LINKX & 35.67 & 35.08 & -0.59 & 0.245")
    _assert_contains("paper/tables/geom_gcn_large_paired_tests.tex", "chameleon & H2GCN & 64.21 & 55.90 & -8.31 & $<0.001$")
    _assert_contains("paper/tables/geom_gcn_large_paired_tests.tex", "squirrel & LINKX & 45.48 & 36.66 & -8.82 & $<0.001$")
    return ["geom_gcn_large: reported means, stds, paired tests, and tables verified"]


def check_webkb() -> list[str]:
    df = _read_csv("results/webkb.csv")
    metrics = _metric_table(df)
    expected = {
        ("texas", "harp"): (86.49, 4.59, 10),
        ("texas", "h2gcn"): (84.59, 3.13, 10),
        ("wisconsin", "harp"): (87.45, 3.23, 10),
        ("wisconsin", "h2gcn"): (85.49, 4.26, 10),
        ("cornell", "harp"): (74.86, 4.60, 10),
    }
    for key, (mean, std, count) in expected.items():
        actual = metrics.loc[key]
        _assert_close(float(actual["mean"]) * 100.0, mean, f"{key} mean")
        _assert_close(float(actual["std"]) * 100.0, std, f"{key} std")
        if int(actual["count"]) != count:
            raise AssertionError(f"{key} count: expected {count}, got {int(actual['count'])}")

    paired = _read_csv("results/webkb_paired_tests.csv").set_index("dataset")
    paired_expected = {
        "texas": ("mixhop", 1.89, 0.132),
        "wisconsin": ("h2gcn", 1.96, 0.085),
        "cornell": ("mlp", -1.62, 0.405),
    }
    for dataset, (baseline, diff_pp, p_value) in paired_expected.items():
        row = paired.loc[dataset]
        if str(row["baseline_model"]) != baseline:
            raise AssertionError(f"{dataset} paired baseline: expected {baseline}, got {row['baseline_model']}")
        _assert_close(float(row["diff_mean"]) * 100.0, diff_pp, f"{dataset} paired diff")
        _assert_close(float(row["p_value"]), p_value, f"{dataset} paired p", tol=0.0005)

    return ["webkb: reported means, stds, and paired tests verified"]


def check_planetoid() -> list[str]:
    df = _read_csv("results/planetoid_all.csv")
    metrics = _metric_table(df)
    expected = {
        ("cora", "appnp"): (82.92, 0.29, 5),
        ("citeseer", "gcn"): (71.80, 0.27, 5),
        ("pubmed", "sgc"): (77.24, 0.29, 5),
        ("pubmed", "harp"): (63.40, 2.63, 5),
    }
    for key, (mean, std, count) in expected.items():
        actual = metrics.loc[key]
        _assert_close(float(actual["mean"]) * 100.0, mean, f"{key} mean")
        _assert_close(float(actual["std"]) * 100.0, std, f"{key} std")
        if int(actual["count"]) != count:
            raise AssertionError(f"{key} count: expected {count}, got {int(actual['count'])}")
    return ["planetoid_all: reported citation sanity metrics verified"]


def check_harp_select() -> list[str]:
    diagnostics = _read_csv("results/harp_select_diagnostics.csv")
    selected = _read_csv("results/harp_select.csv")
    if len(diagnostics) != 80 or len(selected) != 80:
        raise AssertionError(
            f"HARP-Select expected 80 rows, got diagnostics={len(diagnostics)} selected={len(selected)}"
        )
    if not (diagnostics["selector_z"].astype(float) == 1.96).all():
        raise AssertionError("HARP-Select selector_z must remain fixed at 1.96")
    recomputed = diagnostics["val_diff"].astype(float) > diagnostics["selection_threshold"].astype(float)
    if not recomputed.equals(diagnostics["selected_esep"].astype(bool)):
        raise AssertionError("HARP-Select branch decisions do not match val_diff > selection_threshold")
    expected_esep_splits = {
        "actor": 0,
        "amazon-ratings": 0,
        "chameleon": 10,
        "cornell": 0,
        "roman-empire": 10,
        "squirrel": 10,
        "texas": 0,
        "wisconsin": 0,
    }
    observed_esep_splits = diagnostics.groupby("dataset")["selected_esep"].sum().astype(int).to_dict()
    if observed_esep_splits != expected_esep_splits:
        raise AssertionError(
            f"HARP-Select routing changed: expected {expected_esep_splits}, got {observed_esep_splits}"
        )
    expected_means = {
        "actor": 35.08,
        "amazon-ratings": 45.43,
        "chameleon": 63.42,
        "cornell": 74.86,
        "roman-empire": 78.51,
        "squirrel": 46.48,
        "texas": 86.49,
        "wisconsin": 87.45,
    }
    means = selected.groupby("dataset")["test_acc"].mean()
    for dataset, expected in expected_means.items():
        _assert_close(float(means.loc[dataset]) * 100.0, expected, f"HARP-Select {dataset} mean")

    paired = _read_csv("results/harp_select_paired_tests.csv").set_index("dataset")
    paired_expected = {
        "actor": ("linkx", -0.59, 0.245),
        "chameleon": ("h2gcn", -0.79, 0.395),
        "cornell": ("mlp", -1.62, 0.405),
        "squirrel": ("linkx", 1.01, 0.273),
        "texas": ("mixhop", 1.89, 0.132),
        "wisconsin": ("h2gcn", 1.96, 0.085),
    }
    for dataset, (baseline, diff_pp, p_value) in paired_expected.items():
        row = paired.loc[dataset]
        if str(row["baseline_model"]) != baseline:
            raise AssertionError(
                f"HARP-Select {dataset} baseline: expected {baseline}, got {row['baseline_model']}"
            )
        _assert_close(float(row["diff_mean"]) * 100.0, diff_pp, f"HARP-Select {dataset} diff")
        _assert_close(float(row["p_value"]), p_value, f"HARP-Select {dataset} p", tol=0.0005)

    robust = _read_csv("results/harp_select_robust_tests.csv")
    if (robust["sign_flip_p_holm"].astype(float) < 0.05).any():
        raise AssertionError("HARP-Select has a Holm-adjusted sign-flip result below 0.05")
    _assert_contains("paper/tables/harp_select_results.tex", "chameleon & 55.90")
    _assert_contains("paper/tables/harp_select_paired_tests.tex", "squirrel & LINKX & 45.48 & 46.48 & +1.01 & 0.273")

    sensitivity = _read_csv("results/harp_select_threshold_sensitivity_overall.csv").set_index("z")
    if 1.96 not in sensitivity.index:
        raise AssertionError("HARP-Select threshold sensitivity is missing the manuscript z=1.96 row")
    z196 = sensitivity.loc[1.96]
    if int(z196["esep_selections"]) != 30:
        raise AssertionError(f"HARP-Select z=1.96 should select 30 ESep splits, got {z196['esep_selections']}")
    _assert_close(
        float(z196["macro_mean_test_acc"]) * 100.0,
        64.72,
        "HARP-Select z=1.96 sensitivity macro accuracy",
        tol=0.01,
    )
    _assert_close(
        float(z196["macro_mean_oracle_regret"]) * 100.0,
        0.29,
        "HARP-Select z=1.96 sensitivity oracle regret",
        tol=0.01,
    )
    calibration = _read_csv("results/harp_select_margin_calibration.csv").set_index("margin_bin")
    high_margin = calibration.loc[">=1.96"]
    if int(high_margin["splits"]) != 30:
        raise AssertionError("HARP-Select high-margin calibration bin should contain 30 splits")
    _assert_close(
        float(high_margin["esep_test_win_rate"]),
        1.0,
        "HARP-Select high-margin ESep test-win rate",
        tol=1e-12,
    )
    _assert_contains("paper/HARP_SELECTOR_SENSITIVITY.md", "It is not used to retune the paper threshold")
    _assert_contains("paper/tables/harp_select_threshold_sensitivity.tex", "1.96 & 30/80")

    cost = _read_csv("results/harp_select_training_cost.csv").set_index("dataset")
    if len(cost) != 8:
        raise AssertionError(f"HARP-Select cost summary should contain 8 datasets, got {len(cost)}")
    _assert_close(
        float(cost.loc["amazon-ratings", "two_expert_elapsed_mean_sec"]),
        298.92,
        "HARP-Select Amazon-Ratings two-expert seconds",
        tol=0.01,
    )
    _assert_close(
        float(cost.loc["chameleon", "overhead_vs_harp"]),
        2.38,
        "HARP-Select Chameleon overhead",
        tol=0.01,
    )
    macro_two_expert = float(cost["two_expert_elapsed_mean_sec"].mean())
    macro_overhead = float(cost["overhead_vs_harp"].mean())
    _assert_close(macro_two_expert, 94.96, "HARP-Select macro two-expert seconds", tol=0.01)
    _assert_close(macro_overhead, 1.72, "HARP-Select macro overhead", tol=0.01)
    _assert_contains("paper/HARP_SELECTOR_COST.md", "Macro mean overhead versus self-loop HARP: 1.72x")
    _assert_contains("paper/tables/harp_select_training_cost.tex", "amazon-ratings & 182.7 & 116.2 & 298.9")
    return [
        "HARP-Select: validation-only routing, eight-dataset means, paired tests, robust tests, threshold sensitivity, and training cost verified"
    ]


def check_critical_heterophily() -> list[str]:
    df = _read_csv("results/critical_heterophily_harp.csv")
    if len(df) != 40:
        raise AssertionError(f"critical_heterophily_harp.csv: expected 40 rows, got {len(df)}")
    metrics = _metric_table(df)
    expected = {
        ("roman-empire", "harp"): (75.76, 0.63, 10),
        ("roman-empire", "harp_esep"): (78.51, 0.57, 10),
        ("amazon-ratings", "harp"): (45.43, 0.71, 10),
        ("amazon-ratings", "harp_esep"): (45.94, 0.48, 10),
    }
    for key, (mean, std, count) in expected.items():
        actual = metrics.loc[key]
        _assert_close(float(actual["mean"]) * 100.0, mean, f"{key} mean")
        _assert_close(float(actual["std"]) * 100.0, std, f"{key} std")
        if int(actual["count"]) != count:
            raise AssertionError(f"{key} count: expected {count}, got {int(actual['count'])}")
    metadata_expected = {
        "roman-empire": (22662, 32927, 0.0469),
        "amazon-ratings": (24492, 93050, 0.3804),
    }
    for dataset, (nodes, edges, homophily) in metadata_expected.items():
        row = df[df["dataset"] == dataset].iloc[0]
        if int(row["meta_num_nodes"]) != nodes or int(row["meta_num_edges"]) != edges:
            raise AssertionError(
                f"{dataset} metadata mismatch: nodes={row['meta_num_nodes']} edges={row['meta_num_edges']}"
            )
        _assert_close(
            float(row["meta_edge_homophily"]),
            homophily,
            f"{dataset} edge homophily",
            tol=0.0001,
        )

    paired = _read_csv("results/critical_heterophily_harp_esep_paired_tests.csv").set_index("dataset")
    roman = paired.loc["roman-empire"]
    _assert_close(float(roman["diff_mean"]) * 100.0, 2.75, "Roman-Empire HARP-ESep diff")
    if not float(roman["p_value"]) < 0.001:
        raise AssertionError(f"Roman-Empire paired p should be <0.001, got {roman['p_value']}")
    amazon = paired.loc["amazon-ratings"]
    _assert_close(float(amazon["diff_mean"]) * 100.0, 0.50, "Amazon-Ratings HARP-ESep diff")
    _assert_close(float(amazon["p_value"]), 0.172, "Amazon-Ratings paired p", tol=0.0005)

    robust = _read_csv("results/critical_heterophily_harp_esep_robust_tests.csv").set_index("dataset")
    if not float(robust.loc["roman-empire", "sign_flip_p_holm"]) < 0.05:
        raise AssertionError("Roman-Empire exact sign-flip result should remain significant after Holm correction")
    if float(robust.loc["amazon-ratings", "sign_flip_p_holm"]) < 0.05:
        raise AssertionError("Amazon-Ratings exact sign-flip result should remain non-significant")
    return [
        "critical heterophily: external dataset sizes, branch means, paired tests, and robust tests verified"
    ]


def check_critical_external_baselines() -> list[str]:
    baselines = _read_csv("results/critical_heterophily_baselines.csv")
    if len(baselines) != 160:
        raise AssertionError(f"critical_heterophily_baselines.csv: expected 160 rows, got {len(baselines)}")

    summary = _read_csv("results/critical_heterophily_external_summary.csv").set_index(["dataset", "model"])
    expected = {
        ("roman-empire", "h2gcn"): (77.73, 0.68, 10),
        ("roman-empire", "mixhop"): (77.22, 0.50, 10),
        ("roman-empire", "harp_esep"): (78.51, 0.57, 10),
        ("roman-empire", "harp_select"): (78.51, 0.57, 10),
        ("amazon-ratings", "linkx"): (46.34, 0.81, 10),
        ("amazon-ratings", "harp_esep"): (45.94, 0.48, 10),
        ("amazon-ratings", "harp_select"): (45.43, 0.71, 10),
    }
    for key, (mean, std, count) in expected.items():
        actual = summary.loc[key]
        _assert_close(float(actual["mean"]) * 100.0, mean, f"{key} external mean")
        _assert_close(float(actual["std"]) * 100.0, std, f"{key} external std")
        if int(actual["count"]) != count:
            raise AssertionError(f"{key} external count: expected {count}, got {int(actual['count'])}")

    tests = _read_csv("results/critical_heterophily_external_best_baseline_tests.csv").set_index(
        ["dataset", "target_model"]
    )
    roman = tests.loc[("roman-empire", "harp_select")]
    if str(roman["baseline_model"]) != "h2gcn":
        raise AssertionError("Roman-Empire strongest external baseline should be H2GCN")
    _assert_close(float(roman["diff_mean"]) * 100.0, 0.79, "Roman-Empire HARP-Select vs H2GCN diff")
    _assert_close(float(roman["sign_flip_p_holm"]), 0.0078125, "Roman-Empire external Holm sign-flip", tol=1e-9)

    amazon = tests.loc[("amazon-ratings", "harp_select")]
    if str(amazon["baseline_model"]) != "linkx":
        raise AssertionError("Amazon-Ratings strongest external baseline should be LINKX")
    _assert_close(float(amazon["diff_mean"]) * 100.0, -0.91, "Amazon-Ratings HARP-Select vs LINKX diff")
    _assert_close(float(amazon["sign_flip_p_holm"]), 0.01953125, "Amazon-Ratings external Holm sign-flip", tol=1e-9)

    _assert_contains("paper/tables/critical_heterophily_external_main.tex", "amazon-ratings & LINKX")
    _assert_contains("paper/tables/critical_heterophily_external_main.tex", "roman-empire & H2GCN")
    _assert_contains("paper/main.tex", "trails LINKX by 0.91 points")
    return [
        "critical heterophily: external non-HARP baselines, compact table, and best-baseline tests verified"
    ]


def check_binary_critical_heterophily() -> list[str]:
    df = _read_csv("results/critical_heterophily_binary_minesweeper.csv")
    if len(df) != 20:
        raise AssertionError(
            f"critical_heterophily_binary_minesweeper.csv: expected 20 rows, got {len(df)}"
        )
    if set(df["dataset"].astype(str)) != {"minesweeper"}:
        raise AssertionError("Binary critical-heterophily main branch file should contain Minesweeper only")
    if set(df["metric_name"].astype(str)) != {"roc_auc"}:
        raise AssertionError("Minesweeper binary branch comparison must use ROC-AUC")
    if "test_roc_auc" not in df.columns:
        raise AssertionError("Minesweeper binary branch comparison is missing test_roc_auc")
    max_metric_gap = (df["test_acc"].astype(float) - df["test_roc_auc"].astype(float)).abs().max()
    if float(max_metric_gap) > 1e-12:
        raise AssertionError("Minesweeper test_acc compatibility column differs from test_roc_auc")

    metrics = _metric_table(df)
    expected = {
        ("minesweeper", "harp"): (88.18, 0.84, 10),
        ("minesweeper", "harp_esep"): (89.64, 0.67, 10),
    }
    for key, (mean, std, count) in expected.items():
        actual = metrics.loc[key]
        _assert_close(float(actual["mean"]) * 100.0, mean, f"{key} ROC-AUC mean")
        _assert_close(float(actual["std"]) * 100.0, std, f"{key} ROC-AUC std")
        if int(actual["count"]) != count:
            raise AssertionError(f"{key} count: expected {count}, got {int(actual['count'])}")

    paired = _read_csv("results/critical_heterophily_binary_minesweeper_paired_tests.csv").set_index("dataset")
    row = paired.loc["minesweeper"]
    if str(row["target_model"]) != "harp_esep" or str(row["baseline_model"]) != "harp":
        raise AssertionError(
            "Minesweeper paired comparison must be HARP-ESep versus HARP-GNN"
        )
    _assert_close(float(row["target_mean"]) * 100.0, 89.64, "Minesweeper HARP-ESep ROC-AUC")
    _assert_close(float(row["baseline_mean"]) * 100.0, 88.18, "Minesweeper HARP-GNN ROC-AUC")
    _assert_close(float(row["diff_mean"]) * 100.0, 1.45, "Minesweeper ROC-AUC paired diff")
    if not float(row["p_value"]) < 0.001:
        raise AssertionError(f"Minesweeper paired p should be <0.001, got {row['p_value']}")

    robust = _read_csv("results/critical_heterophily_binary_minesweeper_robust_tests.csv").set_index("dataset")
    robust_row = robust.loc["minesweeper"]
    if int(robust_row["wins"]) != 10 or int(robust_row["ties"]) != 0 or int(robust_row["losses"]) != 0:
        raise AssertionError("Minesweeper robust W/T/L should remain 10/0/0")
    _assert_close(
        float(robust_row["bootstrap_ci95_low"]) * 100.0,
        1.27,
        "Minesweeper bootstrap CI low",
        tol=0.01,
    )
    _assert_close(
        float(robust_row["bootstrap_ci95_high"]) * 100.0,
        1.67,
        "Minesweeper bootstrap CI high",
        tol=0.01,
    )
    _assert_close(
        float(robust_row["sign_flip_p_holm"]),
        0.002,
        "Minesweeper Holm sign-flip p",
        tol=0.0001,
    )
    _assert_contains(
        "paper/tables/critical_heterophily_binary_minesweeper_paired_tests.tex",
        "minesweeper & HARP-GNN & 88.18 & 89.64 & +1.45 & $<0.001$",
    )
    _assert_contains(
        "paper/tables/critical_heterophily_binary_minesweeper_robust_tests.tex",
        "minesweeper & HARP-GNN & +1.45 & [+1.27, +1.67] & 10/0/0 & 0.002 & 0.002",
    )

    full = _read_csv("results/critical_heterophily_binary_harp.csv")
    if len(full) != 40:
        raise AssertionError(
            f"critical_heterophily_binary_harp.csv: expected 40 complete Minesweeper/Tolokers rows, got {len(full)}"
        )
    if set(full["dataset"].astype(str)) != {"minesweeper", "tolokers"}:
        raise AssertionError("Full binary critical HARP file should contain Minesweeper and Tolokers only")
    if set(full["metric_name"].astype(str)) != {"roc_auc"}:
        raise AssertionError("Full binary critical HARP comparison must use ROC-AUC")
    full_metrics = _metric_table(full)
    full_expected = {
        ("minesweeper", "harp"): (88.18, 0.84, 10),
        ("minesweeper", "harp_esep"): (89.64, 0.67, 10),
        ("tolokers", "harp"): (82.79, 0.83, 10),
        ("tolokers", "harp_esep"): (79.22, 0.70, 10),
    }
    for key, (mean, std, count) in full_expected.items():
        actual = full_metrics.loc[key]
        _assert_close(float(actual["mean"]) * 100.0, mean, f"{key} full binary ROC-AUC mean")
        _assert_close(float(actual["std"]) * 100.0, std, f"{key} full binary ROC-AUC std")
        if int(actual["count"]) != count:
            raise AssertionError(f"{key} full binary count: expected {count}, got {int(actual['count'])}")

    split_key = "meta_split_id" if "meta_split_id" in full.columns and not full["meta_split_id"].isna().all() else "seed"
    complete_datasets = []
    for dataset, part in full.groupby("dataset"):
        target = part[part["model"].astype(str) == "harp_esep"][[split_key]]
        baseline = part[part["model"].astype(str) == "harp"][[split_key]]
        paired_count = len(target.merge(baseline, on=split_key, how="inner"))
        if paired_count == 10:
            complete_datasets.append(str(dataset))

    complete_paired = _read_csv("results/critical_heterophily_binary_complete_paired_tests.csv").set_index("dataset")
    complete_robust = _read_csv("results/critical_heterophily_binary_complete_robust_tests.csv").set_index("dataset")
    if set(complete_paired.index.astype(str)) != set(complete_datasets):
        raise AssertionError(
            "Complete-only binary paired table is out of sync with 10-split full-run datasets: "
            f"table={sorted(complete_paired.index.astype(str))}, expected={sorted(complete_datasets)}"
        )
    if set(complete_robust.index.astype(str)) != set(complete_datasets):
        raise AssertionError(
            "Complete-only binary robust table is out of sync with 10-split full-run datasets: "
            f"table={sorted(complete_robust.index.astype(str))}, expected={sorted(complete_datasets)}"
        )
    complete_row = complete_paired.loc["minesweeper"]
    _assert_close(float(complete_row["diff_mean"]) * 100.0, 1.45, "Complete-only Minesweeper diff")
    tolokers_row = complete_paired.loc["tolokers"]
    _assert_close(float(tolokers_row["baseline_mean"]) * 100.0, 82.79, "Complete-only Tolokers HARP-GNN ROC-AUC")
    _assert_close(float(tolokers_row["target_mean"]) * 100.0, 79.22, "Complete-only Tolokers HARP-ESep ROC-AUC")
    _assert_close(float(tolokers_row["diff_mean"]) * 100.0, -3.56, "Complete-only Tolokers diff")
    tolokers_robust = complete_robust.loc["tolokers"]
    if int(tolokers_robust["wins"]) != 0 or int(tolokers_robust["ties"]) != 0 or int(tolokers_robust["losses"]) != 10:
        raise AssertionError("Tolokers complete robust W/T/L should remain 0/0/10")
    _assert_close(
        float(tolokers_robust["bootstrap_ci95_low"]) * 100.0,
        -3.96,
        "Tolokers bootstrap CI low",
        tol=0.01,
    )
    _assert_close(
        float(tolokers_robust["bootstrap_ci95_high"]) * 100.0,
        -3.19,
        "Tolokers bootstrap CI high",
        tol=0.01,
    )
    _assert_close(
        float(tolokers_robust["sign_flip_p_holm"]),
        0.00390625,
        "Tolokers Holm sign-flip p",
        tol=1e-9,
    )
    _assert_contains(
        "paper/tables/critical_heterophily_binary_complete_paired_tests.tex",
        "minesweeper & 88.18 $\\pm$ 0.84 & 89.64 $\\pm$ 0.67 & +1.45 & 10 & $<0.001$",
    )
    _assert_contains(
        "paper/tables/critical_heterophily_binary_complete_paired_tests.tex",
        "tolokers & 82.79 $\\pm$ 0.83 & 79.22 $\\pm$ 0.70 & -3.56 & 10 & $<0.001$",
    )
    _assert_contains(
        "paper/tables/critical_heterophily_binary_complete_robust_tests.tex",
        "minesweeper & +1.45 & [+1.27, +1.67] & 10/0/0 & 0.004",
    )
    _assert_contains(
        "paper/tables/critical_heterophily_binary_complete_robust_tests.tex",
        "tolokers & -3.56 & [-3.96, -3.19] & 0/0/10 & 0.004",
    )
    _assert_contains("paper/main.tex", "The binary critical-heterophily rows in Table~\\ref{tab:binary-critical}")
    _assert_contains("paper/main.tex", "On Tolokers, the direction reverses")
    _assert_contains(
        "paper/main.tex",
        "validation selector uses an accuracy-derived uncertainty rule and is therefore not applied to ROC-AUC datasets",
    )
    return [
        "binary critical heterophily: Minesweeper/Tolokers ROC-AUC means, paired tests, robust tests, complete-only tables, and manuscript text verified"
    ]


def check_manuscript_language() -> list[str]:
    banned = [
        "current reproducible draft",
        "Preliminary observations",
        "Planetoid citation sanity-check results",
        "Preliminary synthetic accuracy",
        "HARP-GNNlearns",
        "HARP-GNNconstructs",
    ]
    text = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    found = [phrase for phrase in banned if phrase in text]
    if found:
        raise AssertionError("Manuscript still contains draft/proofing language: " + ", ".join(found))
    _assert_contains("paper/main.tex", "Main Results")
    _assert_contains("paper/main.tex", "evidence is deliberately bounded")
    return ["paper/main.tex: manuscript language smoke check passed"]


def run_checks() -> list[str]:
    messages: list[str] = []
    for config in [
        "configs/geom_gcn_large.yaml",
        "configs/critical_heterophily_baselines.yaml",
        "configs/critical_heterophily_harp.yaml",
        "configs/critical_heterophily_binary_harp.yaml",
        "configs/critical_heterophily_binary_smoke.yaml",
        "configs/webkb.yaml",
        "configs/webkb_harp_esep.yaml",
        "configs/planetoid_all.yaml",
    ]:
        messages.extend(check_config_coverage(config))
    messages.extend(check_result_audit())
    messages.extend(check_geom_gcn_large())
    messages.extend(check_webkb())
    messages.extend(check_planetoid())
    messages.extend(check_critical_heterophily())
    messages.extend(check_critical_external_baselines())
    messages.extend(check_binary_critical_heterophily())
    messages.extend(check_harp_select())
    messages.extend(check_manuscript_language())
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify core reported results against CSV outputs and generated tables.")
    parser.add_argument("--quiet", action="store_true", help="Only print failures.")
    args = parser.parse_args()

    try:
        messages = run_checks()
    except AssertionError as exc:
        raise SystemExit(f"[failed] {exc}") from exc

    if not args.quiet:
        for message in messages:
            print(f"[ok] {message}")
        print("[ok] reported-result verification passed")


if __name__ == "__main__":
    main()
