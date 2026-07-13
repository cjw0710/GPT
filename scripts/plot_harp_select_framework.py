from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


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


C = {
    "ink": "#202936",
    "muted": "#5f6b76",
    "navy": "#173f73",
    "navy_dark": "#0e2d55",
    "line": "#c9d2da",
    "outer": "#f3f7fb",
    "white": "#ffffff",
    "blue": "#2c67a2",
    "blue_soft": "#e8f1fb",
    "green": "#3b805b",
    "green_soft": "#e7f3eb",
    "red": "#a14a42",
    "red_soft": "#f8e9e7",
    "gold": "#8b6918",
    "gold_soft": "#fff2c8",
    "purple": "#694f8d",
    "purple_soft": "#f0eafb",
    "warm": "#f8f4ee",
}


def rounded(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str,
    edge: str,
    lw: float = 0.8,
    radius: float = 0.08,
    linestyle: object = "-",
    z: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        linestyle=linestyle,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    face: str,
    edge: str = "#aeb9c3",
    dashed: bool = False,
    title_size: float = 7.2,
) -> None:
    rounded(
        ax,
        x + 0.045,
        y - 0.045,
        w,
        h,
        face="#d9dee4",
        edge="none",
        lw=0,
        radius=0.10,
        z=0,
    )
    rounded(
        ax,
        x,
        y,
        w,
        h,
        face=face,
        edge=edge,
        lw=0.85,
        radius=0.10,
        linestyle=(0, (4, 2.6)) if dashed else "-",
        z=1,
    )
    ax.text(
        x + w / 2,
        y + h - 0.22,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=C["ink"],
        zorder=8,
    )


def badge(ax: plt.Axes, cx: float, cy: float, w: float, label: str, color: str = C["navy"]) -> None:
    height = 0.27
    rounded(ax, cx - w / 2 + 0.025, cy - height / 2 - 0.025, w, height, face="#bcc8d4", edge="none", lw=0, radius=0.05, z=7)
    rounded(ax, cx - w / 2, cy - height / 2, w, height, face=color, edge=C["navy_dark"], lw=0.42, radius=0.05, z=8)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=5.35, fontweight="bold", color=C["white"], zorder=9)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = C["navy_dark"],
    lw: float = 1.0,
    rad: float = 0.0,
    ms: float = 9.0,
    style: object = "-",
    z: int = 9,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            linewidth=lw,
            linestyle=style,
            mutation_scale=ms,
            shrinkA=2,
            shrinkB=2,
            zorder=z,
        )
    )


def graph_icon(
    ax: plt.Axes,
    cx: float,
    cy: float,
    *,
    scale: float,
    color: str = "#6f7e8b",
    self_loops: bool = False,
    monochrome: bool = False,
) -> None:
    nodes = [
        (cx - 0.39 * scale, cy + 0.22 * scale),
        (cx + 0.05 * scale, cy + 0.36 * scale),
        (cx + 0.40 * scale, cy + 0.06 * scale),
        (cx + 0.08 * scale, cy - 0.34 * scale),
        (cx - 0.40 * scale, cy - 0.18 * scale),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 3), (1, 3)]
    for i, j in edges:
        ax.plot(
            [nodes[i][0], nodes[j][0]],
            [nodes[i][1], nodes[j][1]],
            color=color,
            linewidth=0.82,
            alpha=0.78,
            zorder=4,
        )
    fills = ["#dbeafe", "#dcfce7", "#fee2e2", "#ede9fe", "#fef3c7"]
    if monochrome:
        fills = [C["white"]] * len(nodes)
    for idx, ((x, y), fill) in enumerate(zip(nodes, fills)):
        r = 0.080 * scale
        ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=color, linewidth=0.62, zorder=5))
        if self_loops and idx in (1, 3):
            ax.add_patch(Arc((x, y), 0.23 * scale, 0.21 * scale, theta1=20, theta2=325, color=color, lw=0.65, zorder=4))


def ego_view(ax: plt.Axes, cx: float, cy: float, *, scale: float = 1.0) -> None:
    ego_x = cx - 0.40 * scale
    ax.add_patch(Circle((ego_x, cy), 0.12 * scale, facecolor=C["white"], edgecolor=C["green"], linewidth=0.95, zorder=6))
    ax.text(ego_x, cy, "ego", ha="center", va="center", fontsize=4.4 * scale, color=C["green"], fontweight="bold", zorder=7)
    neighbors = [
        (cx + 0.02 * scale, cy + 0.28 * scale),
        (cx + 0.36 * scale, cy + 0.04 * scale),
        (cx + 0.18 * scale, cy - 0.29 * scale),
        (cx - 0.10 * scale, cy - 0.19 * scale),
    ]
    for i, (x0, y0) in enumerate(neighbors):
        x1, y1 = neighbors[(i + 1) % len(neighbors)]
        ax.plot([x0, x1], [y0, y1], color="#8eb8a0", linewidth=0.75, zorder=4)
    for x, y in neighbors:
        ax.add_patch(Circle((x, y), 0.067 * scale, facecolor=C["green_soft"], edgecolor=C["green"], linewidth=0.65, zorder=5))
    ax.plot([ego_x + 0.14 * scale, cx - 0.18 * scale], [cy, cy], color=C["green"], linewidth=0.8, linestyle=(0, (2, 2)), zorder=4)


def filter_plot(ax: plt.Axes, cx: float, cy: float, *, color: str, no_self: bool, scale: float = 1.0) -> None:
    w, h = 1.02 * scale, 0.52 * scale
    rounded(ax, cx - w / 2, cy - h / 2, w, h, face=C["white"], edge=color, lw=0.7, radius=0.035, z=3)
    x0, y0 = cx - 0.38 * scale, cy - 0.14 * scale
    ax.plot([x0, x0], [y0, cy + 0.16 * scale], color=C["line"], lw=0.55, zorder=4)
    ax.plot([x0, cx + 0.39 * scale], [y0, y0], color=C["line"], lw=0.55, zorder=4)
    if no_self:
        ax.plot(
            [cx - 0.34 * scale, cx - 0.06 * scale, cx + 0.16 * scale, cx + 0.34 * scale],
            [cy - 0.02 * scale, cy + 0.07 * scale, cy + 0.17 * scale, cy - 0.02 * scale],
            color=C["green"],
            lw=1.25,
            zorder=6,
        )
        ax.plot([cx - 0.23 * scale, cx - 0.04 * scale], [cy + 0.18 * scale, cy + 0.18 * scale], color=C["green"], lw=1.25, zorder=6)
        ax.plot([cx + 0.08 * scale, cx + 0.27 * scale], [cy + 0.18 * scale, cy + 0.18 * scale], color=C["red"], lw=1.25, zorder=6)
    else:
        ax.plot([cx - 0.33 * scale, cx - 0.08 * scale], [cy + 0.13 * scale, cy + 0.01 * scale], color=C["blue"], lw=1.3, zorder=6)
        ax.plot([cx + 0.07 * scale, cx + 0.34 * scale], [cy - 0.01 * scale, cy + 0.15 * scale], color=C["red"], lw=1.3, zorder=6)


def gate_icon(ax: plt.Axes, cx: float, cy: float, *, scale: float = 1.0) -> None:
    r = 0.24 * scale
    ax.add_patch(Circle((cx - 0.14 * scale, cy), r, facecolor=C["blue_soft"], edgecolor=C["blue"], linewidth=0.85, zorder=4))
    ax.add_patch(Circle((cx + 0.14 * scale, cy), r, facecolor=C["red_soft"], edgecolor=C["red"], linewidth=0.85, alpha=0.95, zorder=4))
    ax.add_patch(Circle((cx, cy), 0.075 * scale, facecolor=C["white"], edgecolor=C["blue"], linewidth=0.8, zorder=6))
    ax.plot([cx, cx + 0.31 * scale], [cy, cy + 0.18 * scale], color=C["blue"], linewidth=1.0, zorder=7)


def fusion_icon(ax: plt.Axes, cx: float, cy: float, *, scale: float = 1.0) -> None:
    w, h = 0.36 * scale, 0.38 * scale
    ax.add_patch(Rectangle((cx - w, cy - h / 2), w, h, facecolor=C["white"], edgecolor=C["green"], linewidth=0.8, zorder=4))
    ax.add_patch(Rectangle((cx, cy - h / 2), w, h, facecolor=C["green_soft"], edgecolor=C["green"], linewidth=0.8, zorder=4))
    ax.text(cx - w / 2, cy, r"$B_0$", ha="center", va="center", fontsize=7.3 * scale, color=C["ink"], zorder=6)
    ax.text(cx + w / 2, cy, r"$U$", ha="center", va="center", fontsize=7.3 * scale, color=C["ink"], zorder=6)
    ax.plot([cx - w - 0.08 * scale, cx - w - 0.13 * scale, cx - w - 0.13 * scale, cx - w - 0.08 * scale], [cy + h / 2 + 0.05 * scale, cy + h / 2 + 0.05 * scale, cy - h / 2 - 0.05 * scale, cy - h / 2 - 0.05 * scale], color=C["green"], lw=0.9, zorder=6)
    ax.plot([cx + w + 0.08 * scale, cx + w + 0.13 * scale, cx + w + 0.13 * scale, cx + w + 0.08 * scale], [cy + h / 2 + 0.05 * scale, cy + h / 2 + 0.05 * scale, cy - h / 2 - 0.05 * scale, cy - h / 2 - 0.05 * scale], color=C["green"], lw=0.9, zorder=6)


def score_meter(ax: plt.Axes, cx: float, cy: float, *, score: str, color: str) -> None:
    rounded(ax, cx - 0.48, cy - 0.27, 0.96, 0.54, face=C["white"], edge=color, lw=0.72, radius=0.035, z=3)
    ax.text(cx - 0.27, cy + 0.12, score, ha="center", va="center", fontsize=7.2, color=color, fontweight="bold", zorder=6)
    ax.plot([cx - 0.10, cx + 0.30], [cy + 0.11, cy + 0.11], color="#d9dfe5", lw=2.4, solid_capstyle="round", zorder=4)
    ax.plot([cx - 0.10, cx + 0.16], [cy + 0.11, cy + 0.11], color=color, lw=2.4, solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((cx + 0.16, cy + 0.11), 0.035, facecolor=C["white"], edgecolor=color, lw=0.7, zorder=6))
    ax.text(cx, cy - 0.13, "val. score", ha="center", va="center", fontsize=5.35, color=C["muted"], zorder=6)


def lock_icon(ax: plt.Axes, cx: float, cy: float, *, scale: float = 1.0) -> None:
    ax.add_patch(Rectangle((cx - 0.17 * scale, cy - 0.18 * scale), 0.34 * scale, 0.29 * scale, facecolor=C["white"], edgecolor=C["purple"], linewidth=1.0, zorder=6))
    ax.add_patch(Arc((cx, cy + 0.11 * scale), 0.29 * scale, 0.31 * scale, theta1=0, theta2=180, color=C["purple"], linewidth=1.05, zorder=6))
    ax.add_patch(Circle((cx, cy - 0.03 * scale), 0.030 * scale, facecolor=C["purple"], edgecolor="none", zorder=7))


def draw_input_panels(ax: plt.Axes) -> None:
    panel(ax, 0.35, 3.25, 2.35, 2.00, title="Fixed Graph Split", face=C["white"], title_size=7.3)
    graph_icon(ax, 1.52, 4.39, scale=0.92)
    ax.text(1.52, 3.75, r"$G=(V,E),\;X;\quad Y_{\rm tr},Y_{\rm val}$", ha="center", va="center", fontsize=5.95, color=C["ink"], zorder=7)
    badge(ax, 1.52, 3.40, 1.26, "FIXED SPLIT")

    panel(ax, 0.35, 0.55, 2.35, 2.35, title="Two Structural Views", face="#f8fbfd", dashed=True, title_size=6.6)
    graph_icon(ax, 0.97, 1.78, scale=0.67, color=C["blue"], self_loops=True, monochrome=True)
    ego_view(ax, 2.02, 1.78, scale=0.85)
    ax.plot([1.52, 1.52], [1.11, 2.38], color=C["line"], lw=0.6, linestyle=(0, (2, 2)), zorder=3)
    ax.text(0.97, 1.15, r"self-loop $\hat A$", ha="center", va="center", fontsize=5.7, color=C["blue"], fontweight="bold", zorder=7)
    ax.text(2.02, 1.15, r"no-self $\tilde A$", ha="center", va="center", fontsize=5.7, color=C["green"], fontweight="bold", zorder=7)
    badge(ax, 1.52, 0.73, 2.00, "STRUCTURAL PRIORS")


def draw_expert_panel(ax: plt.Axes) -> None:
    panel(ax, 3.05, 2.72, 7.45, 2.53, title="Parallel Residual Polynomial Experts", face=C["warm"], title_size=7.7)
    ax.text(6.78, 4.76, "identical labels, masks, and optimization budget", ha="center", va="center", fontsize=5.8, color=C["muted"], zorder=7)

    rounded(ax, 3.25, 3.72, 7.05, 0.88, face=C["blue_soft"], edge=C["blue"], lw=0.82, radius=0.06, z=2)
    rounded(ax, 3.25, 2.82, 7.05, 0.82, face=C["green_soft"], edge=C["green"], lw=0.82, radius=0.06, z=2)

    ax.add_patch(Circle((3.48, 4.39), 0.105, facecolor=C["white"], edgecolor=C["blue"], linewidth=0.85, zorder=7))
    ax.text(3.48, 4.39, "H", ha="center", va="center", fontsize=5.45, color=C["blue"], fontweight="bold", zorder=8)
    ax.text(3.65, 4.39, "HARP-GNN", ha="left", va="center", fontsize=5.9, color=C["blue"], fontweight="bold", zorder=8)

    ax.add_patch(Circle((3.48, 3.48), 0.105, facecolor=C["white"], edgecolor=C["green"], linewidth=0.85, zorder=7))
    ax.text(3.48, 3.48, "E", ha="center", va="center", fontsize=5.45, color=C["green"], fontweight="bold", zorder=8)
    ax.text(3.65, 3.48, "HARP-ESep", ha="left", va="center", fontsize=5.75, color=C["green"], fontweight="bold", zorder=8)

    xs = [4.74, 6.02, 7.34, 9.48]
    top_y, bottom_y = 4.05, 3.13

    graph_icon(ax, xs[0], top_y, scale=0.43, color=C["blue"], self_loops=True, monochrome=True)
    filter_plot(ax, xs[1], top_y, color=C["blue"], no_self=False, scale=0.88)
    gate_icon(ax, xs[2], top_y, scale=0.90)
    score_meter(ax, xs[3], top_y, score=r"$a_H$", color=C["blue"])

    ego_view(ax, xs[0], bottom_y, scale=0.58)
    filter_plot(ax, xs[1], bottom_y, color=C["green"], no_self=True, scale=0.88)
    fusion_icon(ax, xs[2], bottom_y, scale=0.90)
    score_meter(ax, xs[3], bottom_y, score=r"$a_E$", color=C["green"])

    for x0, x1, yy, color in [
        (5.08, 5.52, top_y, C["blue"]),
        (6.49, 6.94, top_y, C["blue"]),
        (7.77, 8.94, top_y, C["blue"]),
        (5.08, 5.52, bottom_y, C["green"]),
        (6.49, 6.94, bottom_y, C["green"]),
        (7.77, 8.94, bottom_y, C["green"]),
    ]:
        arrow(ax, (x0, yy), (x1, yy), color=color, lw=0.9, ms=7.2)

    labels = ["self-loop", "basis bank", "node gate", ""]
    labels_e = ["ego-separated", "basis bank", "ego fusion", ""]
    for xx, label in zip(xs, labels):
        ax.text(xx, 3.78, label, ha="center", va="center", fontsize=5.2, color=C["ink"], zorder=8)
    for xx, label in zip(xs, labels_e):
        ax.text(xx, 2.88, label, ha="center", va="center", fontsize=5.2, color=C["ink"], zorder=8)


def draw_basis_panel(ax: plt.Axes) -> None:
    panel(ax, 3.05, 0.55, 7.45, 1.88, title="Residual Polynomial Basis Zoom", face="#fbfcfd", dashed=True, title_size=7.2)
    ax.plot([6.78, 6.78], [0.83, 2.10], color=C["line"], lw=0.65, linestyle=(0, (2, 2)), zorder=3)

    ax.text(3.42, 1.94, "H: self-loop family", ha="left", va="center", fontsize=5.9, color=C["blue"], fontweight="bold", zorder=7)
    filter_plot(ax, 4.25, 1.39, color=C["blue"], no_self=False, scale=1.0)
    ax.text(5.60, 1.53, r"$L_k=\hat A^kX$", ha="center", va="center", fontsize=6.4, color=C["ink"], zorder=7)
    ax.text(5.60, 1.18, r"$H_k=L_{k-1}-L_k$", ha="center", va="center", fontsize=6.25, color=C["ink"], zorder=7)

    ax.text(7.12, 1.94, "E: no-self family", ha="left", va="center", fontsize=5.9, color=C["green"], fontweight="bold", zorder=7)
    filter_plot(ax, 7.92, 1.39, color=C["green"], no_self=True, scale=1.0)
    ax.text(9.20, 1.53, r"$B_k=\tilde A^kB_0$", ha="center", va="center", fontsize=6.4, color=C["ink"], zorder=7)
    ax.text(9.20, 1.18, r"$[B_0,\,U]$ fusion", ha="center", va="center", fontsize=6.25, color=C["ink"], zorder=7)

    badge(ax, 6.78, 0.73, 1.55, "SPECTRAL BASES")
    arrow(ax, (5.95, 2.82), (5.95, 2.43), color=C["navy"], lw=0.75, ms=6.5, style=(0, (3, 2)))


def draw_router(ax: plt.Axes) -> None:
    panel(ax, 10.82, 3.25, 3.68, 2.00, title="Validation-Calibrated Router", face=C["gold_soft"], edge=C["gold"], title_size=7.1)
    ax.text(12.66, 4.82, "validation labels only", ha="center", va="center", fontsize=5.2, color=C["gold"], zorder=7)
    x0, x1 = 11.72, 13.98
    for yy in (4.55, 4.29):
        ax.plot([x0, x1], [yy, yy], color="#e0d8ba", lw=3.2, solid_capstyle="round", zorder=4)
    ax.text(11.30, 4.55, r"$a_H$", ha="center", va="center", fontsize=6.9, color=C["blue"], fontweight="bold", zorder=7)
    ax.text(11.30, 4.29, r"$a_E$", ha="center", va="center", fontsize=6.9, color=C["green"], fontweight="bold", zorder=7)
    ax.plot([x0, 13.02], [4.55, 4.55], color=C["blue"], lw=3.2, solid_capstyle="round", zorder=5)
    ax.plot([x0, 13.50], [4.29, 4.29], color=C["green"], lw=3.2, solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((13.02, 4.55), 0.045, facecolor=C["white"], edgecolor=C["blue"], lw=0.7, zorder=6))
    ax.add_patch(Circle((13.50, 4.29), 0.045, facecolor=C["white"], edgecolor=C["green"], lw=0.7, zorder=6))

    ax.text(12.66, 4.00, r"$\Delta_{\rm val}=a_E-a_H$", ha="center", va="center", fontsize=7.05, color=C["ink"], zorder=7)
    axis_left, axis_right, axis_y = 11.45, 13.87, 3.61
    tau_x = 12.66
    ax.add_patch(Rectangle((axis_left, axis_y - 0.075), tau_x - axis_left, 0.15, facecolor=C["blue_soft"], edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((tau_x, axis_y - 0.075), axis_right - tau_x, 0.15, facecolor=C["green_soft"], edgecolor="none", zorder=3))
    ax.plot([axis_left, axis_right], [axis_y, axis_y], color=C["gold"], lw=0.9, zorder=5)
    ax.plot([tau_x, tau_x], [axis_y - 0.14, axis_y + 0.16], color=C["gold"], lw=0.8, linestyle=(0, (2, 2)), zorder=6)
    ax.text(12.05, 3.78, "retain H", ha="center", va="center", fontsize=5.3, color=C["blue"], fontweight="bold", zorder=7)
    ax.text(13.28, 3.78, "select E", ha="center", va="center", fontsize=5.3, color=C["green"], fontweight="bold", zorder=7)
    ax.text(12.66, 3.39, r"$\tau=1.96\,\mathrm{SE}$", ha="center", va="center", fontsize=5.8, color=C["gold"], zorder=7)


def draw_test_panel(ax: plt.Axes) -> None:
    panel(ax, 10.82, 0.55, 3.68, 2.35, title="Locked Test Evaluation", face=C["purple_soft"], edge=C["purple"], title_size=7.2)
    token_y = 2.28
    for xx, label, color, soft in [
        (11.78, "H", C["blue"], C["blue_soft"]),
        (13.52, "E", C["green"], C["green_soft"]),
    ]:
        ax.add_patch(Circle((xx, token_y), 0.15, facecolor=soft, edgecolor=color, linewidth=0.9, zorder=6))
        ax.text(xx, token_y, label, ha="center", va="center", fontsize=6.5, color=color, fontweight="bold", zorder=7)
        arrow(ax, (xx, token_y - 0.15), (12.65, 2.00), color=C["purple"], lw=0.7, ms=6.5)

    ax.add_patch(Circle((12.65, 1.96), 0.12, facecolor=C["white"], edgecolor=C["purple"], linewidth=0.85, zorder=7))
    ax.text(12.65, 1.96, "1", ha="center", va="center", fontsize=5.7, color=C["purple"], fontweight="bold", zorder=8)
    arrow(ax, (12.65, 1.83), (12.65, 1.67), color=C["purple"], lw=0.8, ms=6.0)
    lock_icon(ax, 12.65, 1.53, scale=0.92)

    rounded(ax, 11.45, 0.96, 2.40, 0.40, face=C["white"], edge=C["purple"], lw=0.65, radius=0.04, z=3)
    ax.text(12.65, 1.16, "test score  +  oracle regret", ha="center", va="center", fontsize=5.55, color=C["ink"], zorder=7)

    badge(ax, 12.65, 0.73, 2.62, "TEST LABELS NEVER ROUTE", color=C["red"])


def plot_framework(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.35, 3.07), dpi=450)
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0.25, 6.25)
    ax.axis("off")

    rounded(ax, 0.12, 0.30, 14.76, 5.82, face=C["outer"], edge="#9eacb8", lw=0.85, radius=0.14, z=0)
    ax.text(0.58, 5.82, "HARP-Select:", ha="left", va="center", fontsize=9.5, fontweight="bold", color=C["navy_dark"], zorder=9)
    ax.text(
        3.10,
        5.82,
        "Validation-Calibrated Routing between Residual Graph Experts",
        ha="left",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=C["ink"],
        zorder=9,
    )

    draw_input_panels(ax)
    draw_expert_panel(ax)
    draw_basis_panel(ax)
    draw_router(ax)
    draw_test_panel(ax)

    arrow(ax, (2.70, 4.72), (3.08, 4.72), color=C["navy_dark"], lw=1.05, ms=8.8)
    arrow(ax, (2.68, 2.20), (3.27, 4.05), color=C["blue"], lw=0.82, rad=-0.18, ms=7.2)
    arrow(ax, (2.68, 1.43), (3.27, 3.13), color=C["green"], lw=0.82, rad=-0.17, ms=7.2)

    arrow(ax, (10.30, 4.05), (10.84, 4.45), color=C["blue"], lw=1.0, rad=-0.10, ms=8.5)
    arrow(ax, (10.30, 3.10), (10.84, 4.13), color=C["green"], lw=1.0, rad=-0.14, ms=8.5)
    arrow(ax, (14.18, 3.25), (14.18, 2.92), color=C["purple"], lw=1.05, ms=8.0)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.035)
    vector_path = output_path.with_suffix(".pdf")
    fig.savefig(vector_path, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"[saved] {output_path}")
    print(f"[saved] {vector_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the HARP-Select framework figure.")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="Output PNG path.")
    args = parser.parse_args()
    plot_framework(Path(args.output))


if __name__ == "__main__":
    main()
