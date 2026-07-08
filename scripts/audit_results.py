from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _name(spec: str | dict[str, Any]) -> str:
    if isinstance(spec, str):
        return spec
    return str(spec["name"])


def _project_root(config_path: Path) -> Path:
    config_path = config_path.resolve()
    if config_path.parent.name == "configs":
        return config_path.parents[1]
    return config_path.parent


def _escape_latex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _safe_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def expected_keys(config_path: Path) -> tuple[set[tuple[str, str, int]], Path]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    datasets = [_name(spec) for spec in cfg["datasets"]]
    models = [_name(spec) for spec in cfg["models"]]
    seeds = [int(seed) for seed in cfg.get("seeds", [0])]
    output_path = _project_root(config_path) / cfg.get("output", "results/results.csv")
    keys = {(dataset, model, seed) for dataset in datasets for model in models for seed in seeds}
    return keys, output_path


def observed_keys(results_path: Path) -> tuple[set[tuple[str, str, int]], int]:
    if not results_path.exists():
        return set(), 0
    df = pd.read_csv(results_path)
    required = {"dataset", "model", "seed"}
    if not required.issubset(df.columns):
        return set(), 0
    keys = {
        (str(row.dataset), str(row.model), int(row.seed))
        for row in df[["dataset", "model", "seed"]].itertuples(index=False)
    }
    duplicate_count = int(df.duplicated(subset=["dataset", "model", "seed"], keep=False).sum())
    return keys, duplicate_count


def audit_config(config_path: Path, root: Path) -> dict[str, object]:
    expected, results_path = expected_keys(config_path)
    observed, duplicate_rows = observed_keys(results_path)
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if not results_path.exists():
        status = "missing-output"
    elif missing:
        status = "incomplete"
    elif extra or duplicate_rows:
        status = "check-extra"
    else:
        status = "complete"
    return {
        "config": _safe_relative(config_path, root),
        "output": _safe_relative(results_path, root),
        "status": status,
        "expected": len(expected),
        "observed": len(expected.intersection(observed)),
        "missing": len(missing),
        "extra": len(extra),
        "duplicate_rows": duplicate_rows,
    }


def write_latex(df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Config & Status & Expected & Observed & Missing & Extra & Dup. \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{_escape_latex(row.config)} & {_escape_latex(row.status)} & "
            f"{row.expected} & {row.observed} & {row.missing} & {row.extra} & {row.duplicate_rows} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit result coverage for experiment configs.")
    parser.add_argument("--config-dir", default="configs", help="Directory containing YAML configs.")
    parser.add_argument("--output-csv", default="results/result_audit.csv", help="CSV audit output.")
    parser.add_argument("--output-tex", default="paper/tables/result_audit.tex", help="LaTeX audit table output.")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    config_dir = Path(args.config_dir)
    rows = [audit_config(path, root) for path in sorted(config_dir.glob("*.yaml"))]
    df = pd.DataFrame(rows).sort_values(["status", "config"])

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    write_latex(df, Path(args.output_tex))

    print(df.to_string(index=False))
    print(f"[saved] {output_csv}")
    print(f"[saved] {args.output_tex}")


if __name__ == "__main__":
    main()
