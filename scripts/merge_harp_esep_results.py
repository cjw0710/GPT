from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_results(base_path: Path, esep_path: Path, output_path: Path) -> None:
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base result CSV: {base_path}")
    if not esep_path.exists():
        raise FileNotFoundError(f"Missing HARP-ESep result CSV: {esep_path}")

    base = pd.read_csv(base_path)
    esep = pd.read_csv(esep_path)
    required = {"dataset", "model", "seed", "test_acc"}
    for path, frame in [(base_path, base), (esep_path, esep)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    merged = pd.concat([base, esep], ignore_index=True)
    duplicate_rows = merged.duplicated(subset=["dataset", "model", "seed"], keep=False)
    if duplicate_rows.any():
        duplicates = merged.loc[duplicate_rows, ["dataset", "model", "seed"]]
        raise ValueError("Merged result CSV contains duplicate dataset/model/seed rows:\n" + duplicates.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"[saved] {output_path} ({len(merged)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge full Geom-GCN results with the HARP-ESep candidate run.")
    parser.add_argument("--base", default="results/geom_gcn_large.csv", help="Base Geom-GCN result CSV.")
    parser.add_argument("--esep", default="results/geom_gcn_harp_esep.csv", help="HARP-ESep result CSV.")
    parser.add_argument(
        "--output",
        default="results/geom_gcn_large_with_harp_esep.csv",
        help="Merged output CSV.",
    )
    args = parser.parse_args()
    merge_results(Path(args.base), Path(args.esep), Path(args.output))


if __name__ == "__main__":
    main()
