from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "harp_select_threshold_sensitivity_overall.csv"
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "selector_sensitivity.png"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


def plot_sensitivity(input_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(input_path).sort_values("z")
    required = {"z", "splits", "esep_selections", "macro_mean_oracle_regret"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing sensitivity columns: {sorted(missing)}")
    if 1.96 not in frame["z"].tolist():
        raise ValueError("The frozen z=1.96 operating point is missing.")

    z = frame["z"].to_numpy()
    regret_pp = 100.0 * frame["macro_mean_oracle_regret"].to_numpy()
    selection_rate = 100.0 * (
        frame["esep_selections"].to_numpy() / frame["splits"].to_numpy()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, left = plt.subplots(figsize=(3.35, 1.75), dpi=400)
    left.set_facecolor("#f7f8fa")
    right = left.twinx()

    left.plot(
        z,
        regret_pp,
        color="#93463f",
        marker="o",
        markersize=3.4,
        linewidth=1.05,
        label="oracle regret",
        zorder=4,
    )
    right.plot(
        z,
        selection_rate,
        color="#2f5f99",
        marker="s",
        markersize=3.1,
        linewidth=1.0,
        label="ESep selection rate",
        zorder=3,
    )
    left.axvline(1.96, color="#8a6b1f", linewidth=0.9, linestyle=(0, (3, 2)), zorder=2)
    left.text(
        1.96,
        left.get_ylim()[1] * 0.96,
        "frozen",
        ha="center",
        va="top",
        fontsize=5.7,
        color="#8a6b1f",
        bbox={"facecolor": "#f7f8fa", "edgecolor": "none", "pad": 0.7},
    )

    left.set_title("Frozen-threshold sensitivity", fontsize=7.3, fontweight="bold", pad=4)
    left.set_xlabel(r"confidence multiplier $z$", fontsize=6.4, labelpad=2)
    left.set_ylabel("oracle regret (pp)", fontsize=6.1, color="#93463f", labelpad=2)
    right.set_ylabel("ESep selections (%)", fontsize=6.1, color="#2f5f99", labelpad=3)
    left.set_xticks(z)
    left.set_xticklabels(["0", "0.5", "1", "1.645", "1.96", "2.58"])
    left.tick_params(axis="both", labelsize=5.7, length=2.2, pad=1.5)
    right.tick_params(axis="y", labelsize=5.7, length=2.2, pad=1.5, colors="#2f5f99")
    left.tick_params(axis="y", colors="#93463f")
    left.grid(axis="y", color="#d9dee4", linewidth=0.5, zorder=0)
    left.set_xlim(-0.08, 2.66)
    left.set_ylim(bottom=0)
    right.set_ylim(0, 70)

    lines = left.get_lines()[:1] + right.get_lines()[:1]
    left.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper left",
        fontsize=5.4,
        frameon=False,
        ncol=2,
        handlelength=1.5,
        columnspacing=0.9,
        borderaxespad=0.2,
    )
    for axis in [left, right]:
        axis.spines["top"].set_visible(False)
    left.spines["left"].set_color("#aeb7c0")
    left.spines["bottom"].set_color("#aeb7c0")
    right.spines["right"].set_color("#aeb7c0")

    fig.subplots_adjust(left=0.18, right=0.83, top=0.82, bottom=0.24)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.035)
    vector_path = output_path.with_suffix(".pdf")
    fig.savefig(vector_path, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"[saved] {output_path}")
    print(f"[saved] {vector_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot frozen HARP-Select threshold sensitivity.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    plot_sensitivity(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
