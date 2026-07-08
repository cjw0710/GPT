from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "paper" / "figures" / "harp_select_framework.png"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


COLORS = {
    "ink": "#1f2933",
    "muted": "#5b6770",
    "panel": "#f7f8fa",
    "blue_fill": "#e7f0fb",
    "blue_edge": "#2f5f99",
    "green_fill": "#e7f4ed",
    "green_edge": "#3b7c59",
    "gold_fill": "#fff1c7",
    "gold_edge": "#8a6b1f",
    "purple_fill": "#f0eafa",
    "purple_edge": "#66508a",
    "red_fill": "#f8e8e6",
    "red_edge": "#93463f",
}


def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    body: str = "",
    facecolor: str,
    edgecolor: str,
    title_size: float = 7.4,
    body_size: float = 6.4,
    linewidth: float = 0.9,
    radius: float = 0.045,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    if title:
        ax.text(
            x + w / 2,
            y + h - 0.13,
            title,
            ha="center",
            va="top",
            fontsize=title_size,
            fontweight="bold",
            color=COLORS["ink"],
            zorder=3,
        )
    if body:
        body_y = y + h / 2 - (0.11 if title else 0.0)
        ax.text(
            x + w / 2,
            body_y,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["ink"],
            linespacing=1.06,
            zorder=3,
        )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#333333",
    rad: float = 0.0,
    linewidth: float = 0.95,
    mutation_scale: float = 9.0,
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
            zorder=7,
        )
    )


def _stage_header(ax: plt.Axes, x0: float, x1: float, label: str, subtitle: str) -> None:
    ax.text(
        (x0 + x1) / 2,
        5.08,
        label.upper(),
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        (x0 + x1) / 2,
        4.86,
        subtitle,
        ha="center",
        va="center",
        fontsize=6.9,
        color=COLORS["muted"],
    )


def _graph_icon(
    ax: plt.Axes,
    cx: float,
    cy: float,
    *,
    scale: float = 1.0,
    edgecolor: str = "#9aa5af",
    self_loops: bool = False,
    no_center_fill: bool = False,
) -> None:
    nodes = [
        (cx - 0.33 * scale, cy + 0.23 * scale),
        (cx + 0.08 * scale, cy + 0.33 * scale),
        (cx + 0.33 * scale, cy - 0.02 * scale),
        (cx - 0.06 * scale, cy - 0.25 * scale),
        (cx - 0.42 * scale, cy - 0.12 * scale),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3), (0, 3)]
    for i, j in edges:
        x0, y0 = nodes[i]
        x1, y1 = nodes[j]
        ax.plot([x0, x1], [y0, y1], color=edgecolor, linewidth=0.75, zorder=4)
    node_colors = ["#dbeafe", "#dcfce7", "#fee2e2", "#ede9fe", "#fef3c7"]
    if no_center_fill:
        node_colors[3] = "#ffffff"
    for idx, ((x, y), c) in enumerate(zip(nodes, node_colors)):
        radius = 0.07 * scale if idx != 3 else 0.082 * scale
        ax.add_patch(Circle((x, y), radius, facecolor=c, edgecolor="#4b5563", linewidth=0.55, zorder=5))
        if self_loops and idx in [1, 3]:
            ax.add_patch(Circle((x, y), radius * 1.65, facecolor="none", edgecolor=edgecolor, linewidth=0.55, zorder=4))


def _mask_strip(ax: plt.Axes, x: float, y: float, w: float) -> None:
    colors = [COLORS["blue_edge"], COLORS["gold_edge"], COLORS["purple_edge"]]
    labels = ["train", "val", "test"]
    for i, (color, label) in enumerate(zip(colors, labels)):
        x0 = x + i * w / 3
        ax.add_patch(Rectangle((x0, y), w / 3 - 0.02, 0.08, facecolor=color, edgecolor="none", zorder=5))
        ax.text(x0 + w / 6, y - 0.07, label, ha="center", va="top", fontsize=5.1, color=COLORS["muted"], zorder=5)


def _tiny_filter(ax: plt.Axes, x: float, y: float, w: float, h: float, *, mode: str) -> None:
    ax.plot([x, x + w], [y, y], color="#d2d8df", linewidth=0.55, zorder=5)
    ax.plot([x, x], [y, y + h], color="#d2d8df", linewidth=0.55, zorder=5)
    xs = [x + w * t / 24 for t in range(25)]
    if mode == "low":
        ys = [y + h * (0.82 - 0.52 * (t / 24) ** 1.25) for t in range(25)]
        color = COLORS["blue_edge"]
    elif mode == "residual":
        ys = [y + h * (0.30 + 0.50 * (t / 24) ** 1.2) for t in range(25)]
        color = COLORS["red_edge"]
    else:
        ys = [y + h * (0.28 + 0.50 * (1 - abs(2 * t / 24 - 1))) for t in range(25)]
        color = COLORS["green_edge"]
    ax.plot(xs, ys, color=color, linewidth=1.0, zorder=6)


def _filter_bank_card(ax: plt.Axes, x: float, y: float, edge: str) -> None:
    _rounded_box(
        ax,
        x,
        y,
        1.42,
        0.72,
        title="filter bank",
        facecolor="#ffffff",
        edgecolor=edge,
        title_size=6.5,
    )
    _tiny_filter(ax, x + 0.17, y + 0.15, 0.43, 0.28, mode="low")
    _tiny_filter(ax, x + 0.81, y + 0.15, 0.43, 0.28, mode="residual")
    ax.text(x + 0.38, y + 0.05, "low", ha="center", va="top", fontsize=5.2, color=COLORS["blue_edge"], zorder=6)
    ax.text(x + 1.03, y + 0.05, "res.", ha="center", va="top", fontsize=5.2, color=COLORS["red_edge"], zorder=6)


def _gate_card(ax: plt.Axes, x: float, y: float, edge: str) -> None:
    _rounded_box(
        ax,
        x,
        y,
        1.22,
        0.72,
        title="node gate",
        body=r"$g_i$ blends",
        facecolor="#ffffff",
        edgecolor=edge,
        title_size=6.5,
        body_size=5.9,
    )
    ax.add_patch(Rectangle((x + 0.23, y + 0.13), 0.76, 0.07, facecolor="#e5e7eb", edgecolor="#c7ced6", linewidth=0.35, zorder=5))
    ax.add_patch(Rectangle((x + 0.23, y + 0.13), 0.45, 0.07, facecolor=edge, edgecolor="none", zorder=6))
    ax.add_patch(Circle((x + 0.68, y + 0.165), 0.06, facecolor="#ffffff", edgecolor=edge, linewidth=0.7, zorder=7))


def _score_card(ax: plt.Axes, x: float, y: float, label: str, score: str, edge: str) -> None:
    _rounded_box(
        ax,
        x,
        y,
        0.96,
        0.72,
        title=label,
        body=score + "\nval score",
        facecolor="#ffffff",
        edgecolor=edge,
        title_size=6.2,
        body_size=5.6,
    )


def _router_card(ax: plt.Axes) -> None:
    _rounded_box(
        ax,
        8.86,
        1.42,
        2.34,
        2.24,
        title="Validation evidence router",
        facecolor=COLORS["gold_fill"],
        edgecolor=COLORS["gold_edge"],
        title_size=8.1,
        linewidth=1.0,
    )
    ax.text(
        10.03,
        2.78,
        r"$\Delta_{\mathrm{val}}=a_E-a_H$",
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["ink"],
        zorder=5,
    )
    ax.plot([9.23, 10.83], [2.37, 2.37], color=COLORS["gold_edge"], linewidth=1.0, zorder=5)
    ax.add_patch(Rectangle((9.23, 2.27), 0.75, 0.20, facecolor="#f8e8e6", edgecolor="none", alpha=0.80, zorder=4))
    ax.add_patch(Rectangle((9.98, 2.27), 0.85, 0.20, facecolor="#e7f4ed", edgecolor="none", alpha=0.85, zorder=4))
    ax.plot([9.98, 9.98], [2.20, 2.54], color=COLORS["gold_edge"], linewidth=0.85, linestyle=(0, (2, 2)), zorder=6)
    ax.text(9.98, 2.07, r"$1.96SE$", ha="center", va="top", fontsize=5.8, color=COLORS["gold_edge"], zorder=6)
    ax.add_patch(Circle((10.34, 2.37), 0.07, facecolor=COLORS["green_edge"], edgecolor="#ffffff", linewidth=0.45, zorder=7))
    ax.text(10.03, 1.76, "clear margin selects\notherwise keep safe branch", ha="center", va="center", fontsize=6.25, color=COLORS["ink"], linespacing=1.05, zorder=5)
    _rounded_box(
        ax,
        9.32,
        0.66,
        1.50,
        0.46,
        title="",
        body="rule frozen before test",
        facecolor="#ffffff",
        edgecolor=COLORS["gold_edge"],
        body_size=6.0,
    )


def _locked_card(ax: plt.Axes) -> None:
    _rounded_box(
        ax,
        11.92,
        1.58,
        1.18,
        1.96,
        title="Locked branch",
        body="selected logits\n+ test regret",
        facecolor=COLORS["purple_fill"],
        edgecolor=COLORS["purple_edge"],
        title_size=7.7,
        body_size=6.25,
    )
    # Simple lock icon.
    ax.add_patch(Rectangle((12.36, 2.80), 0.26, 0.23, facecolor="#ffffff", edgecolor=COLORS["purple_edge"], linewidth=0.75, zorder=6))
    ax.add_patch(
        FancyBboxPatch(
            (12.39, 2.98),
            0.20,
            0.18,
            boxstyle="round,pad=0.002,rounding_size=0.055",
            facecolor="none",
            edgecolor=COLORS["purple_edge"],
            linewidth=0.75,
            zorder=6,
        )
    )
    _rounded_box(
        ax,
        11.88,
        0.76,
        1.24,
        0.48,
        title="",
        body="test labels\nnever route",
        facecolor="#ffffff",
        edgecolor=COLORS["purple_edge"],
        body_size=5.85,
    )


def plot_framework(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.20, 3.70), dpi=400)
    ax.set_xlim(0, 13.55)
    ax.set_ylim(0, 5.35)
    ax.axis("off")

    stages = [
        (0.20, 2.05, "fixed split", r"$G,X,Y_{\mathrm{train}},Y_{\mathrm{val}}$"),
        (2.20, 8.45, "parallel expert training", "same split, same budget"),
        (8.60, 11.55, "validation routing", "frozen confidence rule"),
        (11.72, 13.35, "test readout", "after branch lock"),
    ]
    for x0, x1, label, subtitle in stages:
        ax.add_patch(
            Rectangle(
                (x0, 0.25),
                x1 - x0,
                4.45,
                facecolor=COLORS["panel"],
                edgecolor="#e2e6ea",
                linewidth=0.55,
                zorder=0,
            )
        )
        _stage_header(ax, x0, x1, label, subtitle)

    for x in [2.125, 8.525, 11.635]:
        ax.plot([x, x], [0.37, 4.62], color="#c8d0d8", linewidth=0.75, linestyle=(0, (3, 3)), zorder=1)

    # Input split.
    _rounded_box(
        ax,
        0.43,
        1.36,
        1.38,
        2.36,
        title="Input split",
        body=r"$G=(V,E),\,X$",
        facecolor="#ffffff",
        edgecolor="#4b5563",
        title_size=8.2,
        body_size=6.8,
    )
    _graph_icon(ax, 1.12, 2.98, scale=0.90)
    _mask_strip(ax, 0.67, 1.78, 0.90)

    # Expert lanes.
    lane_specs = [
        (2.43, 2.86, 5.72, 1.34, COLORS["blue_fill"], COLORS["blue_edge"], "Self-loop residual expert"),
        (2.43, 1.04, 5.72, 1.34, COLORS["green_fill"], COLORS["green_edge"], "Ego-separated no-self expert"),
    ]
    for x, y, w, h, face, edge, label in lane_specs:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.055",
                linewidth=0.95,
                edgecolor=edge,
                facecolor=face,
                zorder=1,
            )
        )
        ax.text(x + 0.16, y + h - 0.16, label, ha="left", va="top", fontsize=8.6, fontweight="bold", color=edge, zorder=4)

    # Self-loop lane internals.
    _rounded_box(
        ax,
        2.70,
        3.14,
        1.16,
        0.70,
        title="self-loop graph",
        facecolor="#ffffff",
        edgecolor=COLORS["blue_edge"],
        title_size=6.2,
    )
    _graph_icon(ax, 3.28, 3.38, scale=0.36, edgecolor=COLORS["blue_edge"], self_loops=True)
    _filter_bank_card(ax, 4.12, 3.13, COLORS["blue_edge"])
    _gate_card(ax, 5.82, 3.13, COLORS["blue_edge"])
    _score_card(ax, 7.06, 3.13, "HARP-GNN", r"$a_H$", COLORS["blue_edge"])

    # Ego-separated lane internals.
    _rounded_box(
        ax,
        2.70,
        1.32,
        1.16,
        0.70,
        title="ego view",
        facecolor="#ffffff",
        edgecolor=COLORS["green_edge"],
        title_size=5.9,
    )
    _graph_icon(ax, 3.28, 1.56, scale=0.36, edgecolor=COLORS["green_edge"], no_center_fill=True)
    _rounded_box(
        ax,
        4.12,
        1.32,
        1.42,
        0.70,
        title="no-self propagation",
        body=r"$B_k=\tilde A^kB_0$",
        facecolor="#ffffff",
        edgecolor=COLORS["green_edge"],
        title_size=6.0,
        body_size=6.0,
    )
    _rounded_box(
        ax,
        5.82,
        1.32,
        1.22,
        0.70,
        title="residual readout",
        body=r"$[B_0,U]$",
        facecolor="#ffffff",
        edgecolor=COLORS["green_edge"],
        title_size=6.0,
        body_size=6.1,
    )
    _score_card(ax, 7.06, 1.32, "HARP-ESep", r"$a_E$", COLORS["green_edge"])

    # Router and test evaluation.
    _router_card(ax)
    _locked_card(ax)

    # Arrows.
    _arrow(ax, (1.83, 2.72), (2.44, 3.55), color=COLORS["blue_edge"], rad=0.18, linewidth=1.05)
    _arrow(ax, (1.83, 2.30), (2.44, 1.72), color=COLORS["green_edge"], rad=-0.16, linewidth=1.05)

    for y, color in [(3.48, COLORS["blue_edge"]), (1.66, COLORS["green_edge"])]:
        _arrow(ax, (3.86, y), (4.12, y), color=color, linewidth=1.0)
        _arrow(ax, (5.54, y), (5.82, y), color=color, linewidth=1.0)
        _arrow(ax, (7.04, y), (7.06, y), color=color, linewidth=1.0)

    _arrow(ax, (8.00, 3.48), (8.86, 2.98), color=COLORS["blue_edge"], rad=-0.12, linewidth=1.05)
    _arrow(ax, (8.00, 1.66), (8.86, 2.10), color=COLORS["green_edge"], rad=0.12, linewidth=1.05)
    ax.text(8.30, 3.24, r"$a_H$", fontsize=7.2, color=COLORS["blue_edge"], ha="center", va="center")
    ax.text(8.30, 1.89, r"$a_E$", fontsize=7.2, color=COLORS["green_edge"], ha="center", va="center")

    _arrow(ax, (11.20, 2.54), (11.92, 2.55), color=COLORS["purple_edge"], linewidth=1.05)
    _arrow(ax, (10.08, 1.42), (10.08, 1.12), color=COLORS["gold_edge"], linewidth=0.85, mutation_scale=7)
    _arrow(ax, (12.50, 1.58), (12.50, 1.24), color=COLORS["purple_edge"], linewidth=0.85, mutation_scale=7)

    ax.text(
        5.28,
        4.44,
        "same split, different graph priors",
        ha="center",
        va="center",
        fontsize=6.9,
        color=COLORS["muted"],
    )
    ax.text(10.02, 4.44, "validation selects; test measures", ha="center", va="center", fontsize=6.9, color=COLORS["muted"])

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[saved] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the HARP-Select framework figure.")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="Output PNG path.")
    args = parser.parse_args()
    plot_framework(Path(args.output))


if __name__ == "__main__":
    main()
