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


def expected_rows(config_path: Path) -> set[tuple[str, str, int]]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    datasets = [_name(spec) for spec in cfg["datasets"]]
    models = [_name(spec) for spec in cfg["models"]]
    seeds = [int(seed) for seed in cfg.get("seeds", [0])]
    return {(dataset, model, seed) for dataset in datasets for model in models for seed in seeds}


def resolve_results_path(config_path: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _project_root(config_path) / cfg.get("output", "results/results.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a results CSV covers every dataset/model/seed in a config.")
    parser.add_argument("--config", required=True, help="Experiment config path.")
    parser.add_argument("--results", help="Results CSV path. Defaults to the config output field.")
    parser.add_argument("--allow-missing", action="store_true", help="Report missing rows without returning a failing exit code.")
    args = parser.parse_args()

    config_path = Path(args.config)
    results_path = resolve_results_path(config_path, args.results)
    expected = expected_rows(config_path)

    if not results_path.exists():
        print(f"Config: {config_path}")
        print(f"Results: {results_path}")
        print(f"Expected rows: {len(expected)}")
        print("Observed matching rows: 0")
        print(f"\nResults file does not exist: {results_path}")
        if not args.allow_missing:
            raise SystemExit(1)
        return

    df = pd.read_csv(results_path)
    required_columns = {"dataset", "model", "seed"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise SystemExit(f"Results file is missing required columns: {sorted(missing_columns)}")

    observed = {
        (str(row.dataset), str(row.model), int(row.seed))
        for row in df[["dataset", "model", "seed"]].itertuples(index=False)
    }
    missing = sorted(expected.difference(observed))
    extra = sorted(observed.difference(expected))

    duplicate_mask = df.duplicated(subset=["dataset", "model", "seed"], keep=False)
    duplicates = df.loc[duplicate_mask, ["dataset", "model", "seed"]].drop_duplicates()

    print(f"Config: {config_path}")
    print(f"Results: {results_path}")
    print(f"Expected rows: {len(expected)}")
    print(f"Observed matching rows: {len(expected.intersection(observed))}")

    if missing:
        print("\nMissing rows:")
        for dataset, model, seed in missing:
            print(f"  dataset={dataset} model={model} seed={seed}")
    else:
        print("\nMissing rows: none")

    if extra:
        print("\nExtra rows not described by config:")
        for dataset, model, seed in extra:
            print(f"  dataset={dataset} model={model} seed={seed}")
    else:
        print("\nExtra rows: none")

    if not duplicates.empty:
        print("\nDuplicate rows:")
        for row in duplicates.itertuples(index=False):
            print(f"  dataset={row.dataset} model={row.model} seed={int(row.seed)}")
    else:
        print("\nDuplicate rows: none")

    if missing and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
