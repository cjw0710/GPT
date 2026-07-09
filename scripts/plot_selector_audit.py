from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "harp_select_diagnostics.csv"
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "selector_audit.png"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)

COLORS = {
    "harp": "#2f5f99",
    "esep": "#3b7c59",
    "mean": "#1f2933",
    "zero": "#7b8794",
    "grid": "#d9dee4",
    "panel": "#f7f8fa",
}

LABELS = {
    "actor": "Actor",
    "amazon-ratings": "Amazon-Rat.",
    "chameleon": "Chameleon",
    "cornell": "Cornell",
    "roman-empire": "Roman-Emp.",
    "squirrel": "Squirrel",
    "texas": "Texas",
    "wisconsin": "Wisconsin",
}


def _ordered_datasets(frame: pd.DataFrame) -> list[str]:
    means = (
        frame.assign(test_diff=100.0 * (frame["esep_test_acc"] - frame["harp_test_acc"]))
        .groupby("dataset")["test_diff"]
        .mean()
        .sort_values(ascending=False)
    )
    return means.index.tolist()


def _panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    datasets: list[str],
    *,
    value_column: str,
    title: str,
    xlabel: str,
    show_labels: bool,
) -> None:
    ax.set_facecolor(COLORS["panel"])
    jitter = np.linspace(-0.16, 0.16, 10)

    for row, dataset in enumerate(datasets):
        subset = frame.loc[frame["dataset"] == dataset].sort_values("seed")
        values = subset[value_column].to_numpy(dtype=float)
        offsets = jitter[: len(values)]
        colors = np.where(
            subset["selected_esep"].astype(bool).to_numpy(),
            COLORS["esep"],
            COLORS["harp"],
        )
        ax.scatter(
            values,
            row + offsets,
            c=colors,
            s=13,
            alpha=0.62,
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            [values.mean()],
            [row],
            marker="D",
            s=30,
            facecolor="#ffffff",
            edgecolor=COLORS["mean"],
            linewidth=0.8,
            zorder=5,
        )

    ax.axvline(0.0, color=COLORS["zero"], linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.set_title(title, fontsize=7.1, fontweight="bold", pad=4)
    ax.set_xlabel(xlabel, fontsize=6.3, labelpad=2)
    ax.set_yticks(np.arange(len(datasets)))
    if show_labels:
        ax.set_yticklabels([LABELS.get(item, item) for item in datasets], fontsize=6.2)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="x", labelsize=5.8, length=2.2, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45, zorder=0)
    ax.set_ylim(len(datasets) - 0.48, -0.48)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#aeb7c0")
    ax.spines["bottom"].set_linewidth(0.55)


def plot_selector_audit(input_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(input_path)
    required = {
        "dataset",
        "seed",
        "val_diff",
        "selection_threshold",
        "selected_esep",
        "harp_test_acc",
        "esep_test_acc",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing selector diagnostic columns: {sorted(missing)}")

    counts = frame.groupby("dataset")["seed"].nunique()
    if len(counts) != 8 or not (counts == 10).all():
        raise ValueError(f"Expected eight datasets with ten splits each, got {counts.to_dict()}")

    frame = frame.copy()
    frame["validation_surplus_pp"] = 100.0 * (
        frame["val_diff"] - frame["selection_threshold"]
    )
    frame["test_advantage_pp"] = 100.0 * (
        frame["esep_test_acc"] - frame["harp_test_acc"]
    )
    datasets = _ordered_datasets(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.35, 2.15),
        dpi=400,
        sharey=True,
        gridspec_kw={"wspace": 0.10, "width_ratios": [1.05, 1.0]},
    )
    _panel(
        axes[0],
        frame,
        datasets,
        value_column="validation_surplus_pp",
        title="Validation surplus",
        xlabel=r"$\Delta_{\rm val}-1.96\,{\rm SE}$ (pp)",
        show_labels=True,
    )
    _panel(
        axes[1],
        frame,
        datasets,
        value_column="test_advantage_pp",
        title="Test advantage",
        xlabel=r"ESep $-$ GNN (pp)",
        show_labels=False,
    )

    fig.subplots_adjust(left=0.245, right=0.985, top=0.86, bottom=0.19)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.035)
    vector_path = output_path.with_suffix(".pdf")
    fig.savefig(vector_path, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"[saved] {output_path}")
    print(f"[saved] {vector_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot split-level HARP-Select routing evidence.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    plot_selector_audit(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
