from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .data import load_dataset
from .training import train_one_model
from .utils import ensure_dir, read_yaml, resolve_device


def _model_name(spec: str | Dict[str, Any]) -> str:
    if isinstance(spec, str):
        return spec
    return str(spec["name"])


def _model_params(spec: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(spec, str):
        return {}
    return dict(spec.get("params", {}))


def _completed_keys(rows: list[Dict[str, Any]]) -> set[tuple[str, str, int]]:
    keys = set()
    for row in rows:
        if {"dataset", "model", "seed"}.issubset(row):
            keys.add((str(row["dataset"]), str(row["model"]), int(row["seed"])))
    return keys


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "src" / "harp_gnn").is_dir() and (candidate / "configs").is_dir():
            return candidate
    if len(resolved.parents) > 1:
        return resolved.parents[1]
    return Path.cwd().resolve()


def run_config(config_path: Path, resume: bool = False) -> pd.DataFrame:
    cfg = read_yaml(config_path)
    root = _project_root(config_path)
    data_root = root / cfg.get("data_root", "data")
    output_path = root / cfg.get("output", "results/results.csv")
    ensure_dir(output_path.parent)
    device = resolve_device(str(cfg.get("device", "auto")))

    seeds = [int(s) for s in cfg.get("seeds", [0])]
    training_params = dict(cfg.get("training", {}))
    if resume and output_path.exists():
        rows = pd.read_csv(output_path).to_dict(orient="records")
        completed = _completed_keys(rows)
        print(f"[resume] loaded {len(rows)} rows from {output_path}", flush=True)
    else:
        rows = []
        completed = set()

    for dataset_spec in cfg["datasets"]:
        for seed in seeds:
            dataset = load_dataset(dataset_spec, data_root=data_root, seed=seed, device=device)
            for model_spec in cfg["models"]:
                model_name = _model_name(model_spec)
                key = (dataset.name, model_name, seed)
                if key in completed:
                    print(f"[skip] dataset={dataset.name} seed={seed} model={model_name}", flush=True)
                    continue
                params = _model_params(model_spec)
                print(f"[run] dataset={dataset.name} seed={seed} model={model_name} device={device}", flush=True)
                row = train_one_model(
                    dataset=dataset,
                    model_name=model_name,
                    seed=seed,
                    training_params=training_params,
                    model_params=params,
                )
                rows.append(row)
                completed.add(key)
                pd.DataFrame(rows).to_csv(output_path, index=False)
                print(
                    f"[done] dataset={dataset.name} seed={seed} model={model_name} "
                    f"metric={row.get('metric_name', 'accuracy')} "
                    f"val={row['val_acc']:.4f} test={row['test_acc']:.4f}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[saved] {output_path}", flush=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HARP-GNN experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--resume", action="store_true", help="Skip dataset/model/seed rows already present in the output CSV.")
    args = parser.parse_args()
    run_config(Path(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
