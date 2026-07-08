from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_results import MODEL_LABELS, MODEL_ORDER
from src.harp_gnn.data import load_dataset
from src.harp_gnn.models import build_model
from src.harp_gnn.utils import read_yaml, resolve_device


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model).lower(), str(model))


def model_name(spec: str | dict[str, Any]) -> str:
    if isinstance(spec, str):
        return spec
    return str(spec["name"])


def model_params(spec: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(spec, str):
        return {}
    return dict(spec.get("params", {}))


def count_trainable_params(model) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def format_kparams(mean: float, std: float) -> str:
    return f"{mean / 1000.0:.1f} $\\pm$ {std / 1000.0:.1f}K"


def summarize(config_path: Path, output_path: Path, seed: int) -> pd.DataFrame:
    cfg = read_yaml(config_path)
    root = config_path.resolve().parents[1]
    data_root = root / cfg.get("data_root", "data")
    device = resolve_device(str(cfg.get("device", "auto")))
    training_params = dict(cfg.get("training", {}))
    rows: list[dict[str, object]] = []

    for dataset_spec in cfg["datasets"]:
        dataset = load_dataset(dataset_spec, data_root=data_root, seed=seed, device=device)
        for model_spec in cfg["models"]:
            name = model_name(model_spec)
            params = dict(training_params)
            params.update(model_params(model_spec))
            model = build_model(
                name,
                dataset.num_features,
                dataset.num_classes,
                params,
                num_nodes=int(dataset.metadata["num_nodes"]),
            )
            rows.append(
                {
                    "dataset": dataset.name,
                    "model": name,
                    "parameters": count_trainable_params(model),
                }
            )

    raw = pd.DataFrame(rows)
    grouped = (
        raw.groupby("model")
        .agg(
            param_mean=("parameters", "mean"),
            param_std=("parameters", "std"),
            param_min=("parameters", "min"),
            param_max=("parameters", "max"),
            datasets=("dataset", "nunique"),
        )
        .reset_index()
    )
    grouped["Model"] = grouped["model"].map(model_label)
    grouped["Trainable params"] = grouped.apply(
        lambda row: format_kparams(
            float(row["param_mean"]),
            0.0 if pd.isna(row["param_std"]) else float(row["param_std"]),
        ),
        axis=1,
    )
    grouped["Datasets"] = grouped["datasets"].astype(int)
    order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    grouped["order"] = grouped["Model"].map(lambda value: order.get(value, len(order)))
    grouped = grouped.sort_values(["order", "Model"])
    table = grouped[["Model", "Trainable params", "Datasets"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = table.to_latex(
        index=False,
        escape=False,
        column_format="lcc",
    )
    output_path.write_text(latex, encoding="utf-8")
    print(table)
    print(f"[saved] {output_path}")
    grouped.attrs["raw"] = raw
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize model parameter counts for an experiment config.")
    parser.add_argument("--config", required=True, help="Experiment config with datasets and models.")
    parser.add_argument("--output", required=True, help="Output LaTeX table path.")
    parser.add_argument("--csv-output", default=None, help="Optional summary CSV output path.")
    parser.add_argument("--raw-output", default=None, help="Optional per-dataset raw CSV output path.")
    parser.add_argument("--seed", type=int, default=0, help="Seed/split used to construct datasets.")
    args = parser.parse_args()

    grouped = summarize(Path(args.config), Path(args.output), seed=args.seed)
    if args.csv_output is not None:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(csv_output, index=False)
        print(f"[saved] {csv_output}")
    if args.raw_output is not None:
        raw_output = Path(args.raw_output)
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        grouped.attrs["raw"].to_csv(raw_output, index=False)
        print(f"[saved] {raw_output}")


if __name__ == "__main__":
    main()
