from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.harp_gnn.data import load_dataset
from src.harp_gnn.utils import read_yaml, resolve_device


def format_int(value: int) -> str:
    return f"{int(value):,}"


def format_homophily(value: float) -> str:
    return f"{float(value):.3f}"


def dataset_rows(config_path: Path, seed: int) -> pd.DataFrame:
    cfg = read_yaml(config_path)
    root = config_path.resolve().parents[1]
    data_root = root / cfg.get("data_root", "data")
    device = resolve_device(str(cfg.get("device", "auto")))
    seen: set[str] = set()
    rows: list[dict[str, object]] = []

    for spec in cfg["datasets"]:
        dataset = load_dataset(spec, data_root=data_root, seed=seed, device=device)
        if dataset.name in seen:
            continue
        seen.add(dataset.name)
        rows.append(
            {
                "dataset": dataset.name,
                "nodes": int(dataset.metadata["num_nodes"]),
                "edges": int(dataset.metadata["num_edges"]),
                "features": int(dataset.num_features),
                "classes": int(dataset.num_classes),
                "train": int(dataset.train_idx.numel()),
                "val": int(dataset.val_idx.numel()),
                "test": int(dataset.test_idx.numel()),
                "edge_homophily": float(dataset.metadata["edge_homophily"]),
            }
        )
    return pd.DataFrame(rows)


def write_latex(df: pd.DataFrame, output_path: Path) -> None:
    display = pd.DataFrame(
        {
            "Dataset": df["dataset"],
            "Nodes": df["nodes"].map(format_int),
            "Edges": df["edges"].map(format_int),
            "Feat.": df["features"].map(format_int),
            "Classes": df["classes"].map(format_int),
            "Train": df["train"].map(format_int),
            "Val.": df["val"].map(format_int),
            "Test": df["test"].map(format_int),
            "Hom.": df["edge_homophily"].map(format_homophily),
        }
    )
    latex = display.to_latex(
        index=False,
        escape=False,
        column_format="lrrrrrrrr",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    print(display)
    print(f"[saved] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize datasets used by an experiment config.")
    parser.add_argument("--config", required=True, help="Experiment config with a datasets section.")
    parser.add_argument("--output", required=True, help="Output LaTeX table path.")
    parser.add_argument("--csv-output", default=None, help="Optional raw CSV output path.")
    parser.add_argument("--seed", type=int, default=0, help="Seed/split used for split-size reporting.")
    args = parser.parse_args()

    df = dataset_rows(Path(args.config), seed=args.seed)
    write_latex(df, Path(args.output))
    if args.csv_output is not None:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_output, index=False)
        print(f"[saved] {csv_output}")


if __name__ == "__main__":
    main()
