from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


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
    "muted": "#62717d",
    "panel": "#f7f8fa",
    "blue_fill": "#e7f0fb",
    "blue_edge": "#2f5f99",
    "red_fill": "#f8e8e6",
    "red_edge": "#93463f",
    "green_fill": "#e7f4ed",
    "green_edge": "#3b7c59",
    "gold_fill": "#fff1c7",
    "gold_edge": "#8a6b1f",
    "purple_fill": "#f0eafa",
    "purple_edge": "#66508a",
}


def _rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    title_size: float = 5.00,
    body_size: float = 5.70,
    linewidth: float = 0.78,
    body_fraction: float = 0.43,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.045",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    if title:
        ax.text(
            x + w / 2,
            y + h - 0.14,
            title,
            ha="center",
            va="top",
            fontsize=title_size,
            fontweight="semibold",
            fontfamily="DejaVu Sans",
            color=COLORS["ink"],
            zorder=3,
        )
        body_y = y + h * body_fraction
    else:
        body_y = y + h / 2
    ax.text(
        x + w / 2,
        body_y,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=COLORS["ink"],
        linespacing=1.03,
        zorder=3,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    rad: float = 0.0,
    linewidth: float = 0.72,
    mutation_scale: float = 7.2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3,
            shrinkB=3,
            zorder=5,
        )
    )


def _stage_header(ax: plt.Axes, x0: float, x1: float, label: str) -> None:
    ax.text(
        (x0 + x1) / 2,
        5.35,
        label.upper(),
        ha="center",
        va="center",
        fontsize=5.20,
        fontweight="semibold",
        fontfamily="DejaVu Sans",
        color=COLORS["muted"],
    )


def _graph_icon(ax: plt.Axes, cx: float, cy: float, scale: float = 1.0) -> None:
    nodes = [
        (cx - 0.28 * scale, cy + 0.21 * scale),
        (cx + 0.08 * scale, cy + 0.27 * scale),
        (cx + 0.28 * scale, cy - 0.02 * scale),
        (cx - 0.04 * scale, cy - 0.23 * scale),
        (cx - 0.35 * scale, cy - 0.10 * scale),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]
    for i, j in edges:
        x0, y0 = nodes[i]
        x1, y1 = nodes[j]
        ax.plot([x0, x1], [y0, y1], color="#98a6b3", linewidth=0.58, zorder=4)
    node_colors = ["#dbeafe", "#dcfce7", "#fee2e2", "#ede9fe", "#fef3c7"]
    for (x, y), color in zip(nodes, node_colors):
        ax.add_patch(
            Circle((x, y), 0.058 * scale, facecolor=color, edgecolor="#4b5563", linewidth=0.45, zorder=5)
        )


def _tiny_filter(ax: plt.Axes, x: float, y: float, w: float, h: float, *, mode: str) -> None:
    ax.plot([x, x + w], [y, y], color="#cbd4dd", linewidth=0.42, alpha=0.70, zorder=2.4)
    ax.plot([x, x], [y, y + h], color="#cbd4dd", linewidth=0.42, alpha=0.70, zorder=2.4)
    xs = [x + w * t / 24 for t in range(25)]
    if mode == "low":
        ys = [y + h * (0.82 - 0.55 * (t / 24) ** 1.25) for t in range(25)]
        color = COLORS["blue_edge"]
    elif mode == "high":
        ys = [y + h * (0.25 + 0.56 * (t / 24) ** 1.18) for t in range(25)]
        color = COLORS["red_edge"]
    else:
        ys = [y + h * (0.28 + 0.48 * (1 - abs(2 * t / 24 - 1))) for t in range(25)]
        color = COLORS["green_edge"]
    ax.plot(xs, ys, color=color, linewidth=0.72, alpha=0.72, zorder=2.5)


def _gate_meter(ax: plt.Axes, x: float, y: float, w: float, *, edgecolor: str) -> None:
    ax.add_patch(Rectangle((x, y), w, 0.055, facecolor="#e5e7eb", edgecolor="#c7ced6", linewidth=0.32, zorder=5))
    ax.add_patch(Rectangle((x, y), 0.58 * w, 0.055, facecolor=edgecolor, edgecolor="none", zorder=6))
    ax.add_patch(Circle((x + 0.58 * w, y + 0.027), 0.052, facecolor="#ffffff", edgecolor=edgecolor, linewidth=0.62, zorder=7))


def _fusion_icon(ax: plt.Axes, cx: float, cy: float) -> None:
    ax.add_patch(Circle((cx - 0.09, cy), 0.13, facecolor=COLORS["blue_fill"], edgecolor=COLORS["blue_edge"], linewidth=0.55, zorder=5))
    ax.add_patch(Circle((cx + 0.09, cy), 0.13, facecolor=COLORS["red_fill"], edgecolor=COLORS["red_edge"], linewidth=0.55, zorder=5))
    ax.add_patch(Circle((cx, cy), 0.13, facecolor=COLORS["gold_fill"], edgecolor=COLORS["gold_edge"], linewidth=0.65, alpha=0.92, zorder=6))


def _input_box(ax: plt.Axes) -> None:
    patch = FancyBboxPatch(
        (0.22, 1.50),
        1.38,
        2.38,
        boxstyle="round,pad=0.018,rounding_size=0.045",
        linewidth=0.78,
        edgecolor="#4b5563",
        facecolor="#ffffff",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        0.91,
        3.68,
        "Graph\ninput",
        ha="center",
        va="top",
        fontsize=5.20,
        fontweight="semibold",
        fontfamily="DejaVu Sans",
        color=COLORS["ink"],
        zorder=3,
    )
    _graph_icon(ax, 0.91, 2.77, scale=0.64)
    ax.text(
        0.91,
        1.93,
        r"features $X$" + "\n" + r"adjacency $\hat A$",
        ha="center",
        va="center",
        fontsize=4.65,
        color=COLORS["ink"],
        linespacing=1.05,
        zorder=3,
    )


def plot_framework(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.35, 2.64), dpi=400)
    ax.set_xlim(0, 10.85)
    ax.set_ylim(0, 5.62)
    ax.axis("off")

    stages = [
        (0.10, 1.68, "input"),
        (1.84, 4.72, "bases"),
        (4.88, 7.42, "mix"),
        (7.58, 10.74, "prediction"),
    ]
    for x0, x1, label in stages:
        ax.add_patch(
            Rectangle(
                (x0, 0.18),
                x1 - x0,
                4.92,
                facecolor=COLORS["panel"],
                edgecolor="#e2e7ec",
                linewidth=0.48,
                zorder=0,
            )
        )
        _stage_header(ax, x0, x1, label)

    for x in [1.76, 4.80, 7.50]:
        ax.plot([x, x], [0.30, 4.98], color="#cbd4dd", linewidth=0.55, linestyle=(0, (2.2, 2.2)), zorder=1)

    _input_box(ax)

    _rounded_box(
        ax,
        1.97,
        3.58,
        2.55,
        0.98,
        title="Low-pass basis",
        body=r"$L_k=\hat A^kX$",
        facecolor=COLORS["blue_fill"],
        edgecolor=COLORS["blue_edge"],
        body_size=5.55,
    )
    _tiny_filter(ax, 2.10, 3.69, 0.38, 0.32, mode="low")
    _rounded_box(
        ax,
        1.97,
        2.20,
        2.55,
        0.98,
        title="Residual basis",
        body=r"$H_k=L_{k-1}-L_k$",
        facecolor=COLORS["red_fill"],
        edgecolor=COLORS["red_edge"],
        body_size=5.50,
    )
    _tiny_filter(ax, 2.10, 2.31, 0.36, 0.32, mode="high")
    _rounded_box(
        ax,
        1.97,
        0.82,
        2.55,
        0.98,
        title="Node variation",
        body=r"$r_i=\|x_i-\hat A x_i\|$",
        facecolor=COLORS["green_fill"],
        edgecolor=COLORS["green_edge"],
        body_size=5.25,
    )
    _tiny_filter(ax, 2.10, 0.93, 0.36, 0.32, mode="band")

    _rounded_box(
        ax,
        5.05,
        3.58,
        2.05,
        0.98,
        title="Low mix",
        body=r"$Z_L=\sum_k\alpha_k\phi(L_k)$",
        facecolor="#ffffff",
        edgecolor=COLORS["blue_edge"],
        body_size=4.78,
    )
    _rounded_box(
        ax,
        5.05,
        2.20,
        2.05,
        0.98,
        title="High mix",
        body=r"$Z_H=\sum_k\beta_k\psi(H_k)$",
        facecolor="#ffffff",
        edgecolor=COLORS["red_edge"],
        body_size=4.74,
    )
    _rounded_box(
        ax,
        5.05,
        0.82,
        2.05,
        0.98,
        title="Gate",
        body=r"$g_i=\sigma(\mathrm{MLP}(r_i))$",
        facecolor="#ffffff",
        edgecolor=COLORS["green_edge"],
        body_size=4.48,
    )
    _gate_meter(ax, 5.55, 0.91, 1.05, edgecolor=COLORS["green_edge"])

    _rounded_box(
        ax,
        7.76,
        1.70,
        2.22,
        1.66,
        title="Gated fusion",
        body=r"$Z_i=(1-g_i)Z_{L,i}$" + "\n" + r"$+\,g_iZ_{H,i}$",
        facecolor=COLORS["gold_fill"],
        edgecolor=COLORS["gold_edge"],
        body_size=4.92,
        linewidth=0.88,
        body_fraction=0.59,
    )
    _fusion_icon(ax, 8.87, 2.03)
    _rounded_box(
        ax,
        10.19,
        2.10,
        0.50,
        0.88,
        title="",
        body=r"$\hat y_i$",
        facecolor=COLORS["purple_fill"],
        edgecolor=COLORS["purple_edge"],
        body_size=6.70,
        linewidth=0.82,
    )

    _arrow(ax, (1.60, 2.96), (1.97, 4.07), color=COLORS["blue_edge"], rad=0.15, linewidth=0.78)
    _arrow(ax, (1.60, 2.68), (1.97, 2.69), color=COLORS["red_edge"], linewidth=0.78)
    _arrow(ax, (1.60, 2.40), (1.97, 1.31), color=COLORS["green_edge"], rad=-0.15, linewidth=0.78)

    for y, color in [(4.07, COLORS["blue_edge"]), (2.69, COLORS["red_edge"]), (1.31, COLORS["green_edge"])]:
        _arrow(ax, (4.52, y), (5.05, y), color=color, linewidth=0.72)

    _arrow(ax, (7.10, 4.07), (7.76, 3.03), color=COLORS["blue_edge"], rad=-0.10, linewidth=0.74)
    _arrow(ax, (7.10, 2.69), (7.76, 2.64), color=COLORS["red_edge"], linewidth=0.74)
    _arrow(ax, (7.10, 1.31), (7.76, 2.08), color=COLORS["green_edge"], rad=0.12, linewidth=0.74)
    _arrow(ax, (9.98, 2.54), (10.19, 2.54), color=COLORS["purple_edge"], linewidth=0.76)

    label_box = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.35)
    ax.text(7.39, 3.53, r"$Z_L$", ha="center", va="center", fontsize=5.20, color=COLORS["blue_edge"], bbox=label_box)
    ax.text(7.38, 2.47, r"$Z_H$", ha="center", va="center", fontsize=5.20, color=COLORS["red_edge"], bbox=label_box)
    ax.text(7.35, 1.76, r"$g_i$", ha="center", va="center", fontsize=5.20, color=COLORS["green_edge"], bbox=label_box)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.035)
    vector_path = output_path.with_suffix(".pdf")
    fig.savefig(vector_path, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"[saved] {output_path}")
    print(f"[saved] {vector_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the HARP-GNN internal block diagram.")
    parser.add_argument(
        "--output",
        default="paper/figures/harp_framework.png",
        help="Output image path.",
    )
    args = parser.parse_args()
    plot_framework(Path(args.output))


if __name__ == "__main__":
    main()
