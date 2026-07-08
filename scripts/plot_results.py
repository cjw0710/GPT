from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "sgc": "SGC",
    "appnp": "APPNP",
    "harp": "HARP-GNN",
}


def parse_homophily(dataset: str) -> float:
    match = re.search(r"h(\d+)", dataset)
    if not match:
        raise ValueError(f"Cannot parse homophily from dataset name: {dataset}")
    return float(match.group(1)) / 100.0


def plot_synthetic(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    df = df[df["dataset"].str.contains("synthetic_h", regex=False)].copy()
    df["homophily"] = df["dataset"].map(parse_homophily)
    df["model_label"] = df["model"].map(lambda m: MODEL_LABELS.get(str(m).lower(), str(m)))
    grouped = df.groupby(["homophily", "model_label"])["test_acc"].agg(["mean", "std"]).reset_index()

    order = ["MLP", "GCN", "SGC", "APPNP", "HARP-GNN"]
    plt.figure(figsize=(6.5, 4.0))
    for model in order:
        part = grouped[grouped["model_label"] == model].sort_values("homophily")
        if part.empty:
            continue
        plt.errorbar(
            part["homophily"],
            100.0 * part["mean"],
            yerr=100.0 * part["std"].fillna(0.0),
            marker="o",
            linewidth=2,
            capsize=3,
            label=model,
        )
    plt.xlabel("Target homophily")
    plt.ylabel("Test accuracy (%)")
    plt.xticks([0.2, 0.5, 0.8])
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=240)
    plt.close()
    print(f"[saved] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument("--input", required=True, help="Input CSV.")
    parser.add_argument("--output", required=True, help="Output image path.")
    args = parser.parse_args()
    plot_synthetic(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
