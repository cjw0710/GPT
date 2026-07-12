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
    "ink": "#202a35",
    "muted": "#5e6a75",
    "line": "#cfd6dd",
    "data_bg": "#f5f7f9",
    "train_bg": "#f4f7fb",
    "route_bg": "#fffaf0",
    "test_bg": "#f8f5fc",
    "blue": "#2b63a0",
    "blue_soft": "#e4eef9",
    "green": "#3a7f5a",
    "green_soft": "#e5f2ea",
    "red": "#9a463f",
    "red_soft": "#f7e8e6",
    "gold": "#8c6a17",
    "gold_soft": "#fff0bd",
    "purple": "#684f8d",
    "purple_soft": "#eee8f8",
    "white": "#ffffff",
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
    lw: float = 0.9,
    radius: float = 0.05,
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
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    lw: float = 1.1,
    rad: float = 0.0,
    ms: float = 9.0,
    style: str = "-",
    z: int = 8,
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
            shrinkA=3,
            shrinkB=3,
            zorder=z,
        )
    )


def stage_header(
    ax: plt.Axes,
    x0: float,
    x1: float,
    number: int,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    badge_x = x0 + 0.16
    text_x = x0 + 0.38
    ax.add_patch(Circle((badge_x, 5.02), 0.105, facecolor=color, edgecolor="none", zorder=5))
    ax.text(
        badge_x,
        5.02,
        str(number),
        ha="center",
        va="center",
        fontsize=6.4,
        color=C["white"],
        fontweight="bold",
        zorder=6,
    )
    ax.text(
        text_x,
        5.04,
        title.upper(),
        ha="left",
        va="center",
        fontsize=7.45,
        color=C["ink"],
        fontweight="bold",
        zorder=5,
    )
    ax.text(
        text_x,
        4.79,
        subtitle,
        ha="left",
        va="center",
        fontsize=6.25,
        color=C["muted"],
        zorder=5,
    )


def graph_icon(
    ax: plt.Axes,
    cx: float,
    cy: float,
    *,
    scale: float,
    color: str = "#7d8994",
    self_loops: bool = False,
) -> None:
    nodes = [
        (cx - 0.34 * scale, cy + 0.19 * scale),
        (cx + 0.08 * scale, cy + 0.31 * scale),
        (cx + 0.35 * scale, cy + 0.02 * scale),
        (cx + 0.02 * scale, cy - 0.30 * scale),
        (cx - 0.36 * scale, cy - 0.16 * scale),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 3), (1, 3)]
    for i, j in edges:
        ax.plot(
            [nodes[i][0], nodes[j][0]],
            [nodes[i][1], nodes[j][1]],
            color=color,
            linewidth=0.78,
            alpha=0.76,
            zorder=4,
        )
    fills = ["#dbeafe", "#dcfce7", "#fee2e2", "#ede9fe", "#fef3c7"]
    for idx, ((x, y), fill) in enumerate(zip(nodes, fills)):
        r = 0.073 * scale
        ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=color, linewidth=0.62, zorder=5))
        if self_loops and idx in (1, 3):
            ax.add_patch(Arc((x, y), 0.22 * scale, 0.20 * scale, theta1=20, theta2=325, color=color, lw=0.65, zorder=4))


def split_card(ax: plt.Axes) -> None:
    rounded(ax, 0.40, 1.08, 1.40, 2.92, face=C["white"], edge="#53606c", lw=0.95)
    ax.text(1.10, 3.79, "Fixed split", ha="center", va="center", fontsize=7.4, fontweight="bold", color=C["ink"], zorder=6)
    graph_icon(ax, 1.10, 3.12, scale=0.94)
    ax.text(1.10, 2.40, r"$G=(V,E),\;X$", ha="center", va="center", fontsize=7.2, color=C["ink"], zorder=6)
    ax.text(1.10, 2.14, r"$Y_{\rm tr},\;Y_{\rm val}$", ha="center", va="center", fontsize=7.2, color=C["ink"], zorder=6)

    x0, y0, w = 0.68, 1.55, 0.84
    colors = [C["blue"], C["gold"], C["purple"]]
    labels = ["train", "val", "test"]
    for i, (color, label) in enumerate(zip(colors, labels)):
        left = x0 + i * w / 3
        ax.add_patch(Rectangle((left, y0), w / 3 - 0.018, 0.09, facecolor=color, edgecolor="none", zorder=5))
        ax.text(left + w / 6, y0 - 0.08, label, ha="center", va="top", fontsize=5.15, color=C["muted"], zorder=6)
    ax.text(1.10, 1.22, "shared masks", ha="center", va="center", fontsize=5.7, color=C["muted"], zorder=6)


def lane_header(ax: plt.Axes, x: float, y: float, expert: str, name: str, prior: str, color: str) -> None:
    ax.add_patch(Circle((x, y), 0.145, facecolor=C["white"], edgecolor=color, linewidth=1.0, zorder=6))
    ax.text(x, y, expert, ha="center", va="center", fontsize=6.8, fontweight="bold", color=color, zorder=7)
    ax.text(x + 0.23, y + 0.02, name, ha="left", va="center", fontsize=7.4, fontweight="bold", color=color, zorder=7)
    ax.text(x + 1.76, y + 0.02, prior, ha="left", va="center", fontsize=6.0, color=C["muted"], zorder=7)


def self_prior_icon(ax: plt.Axes, cx: float, cy: float) -> None:
    graph_icon(ax, cx + 0.08, cy + 0.05, scale=0.58, color=C["blue"], self_loops=True)
    ax.add_patch(Circle((cx - 0.40, cy + 0.24), 0.145, facecolor=C["white"], edgecolor=C["blue"], linewidth=0.9, zorder=7))
    ax.text(cx - 0.40, cy + 0.24, r"$+I$", ha="center", va="center", fontsize=7.1, color=C["blue"], fontweight="bold", zorder=8)


def ego_prior_icon(ax: plt.Axes, cx: float, cy: float) -> None:
    ego_x = cx - 0.36
    ax.add_patch(Circle((ego_x, cy + 0.03), 0.12, facecolor=C["white"], edgecolor=C["green"], linewidth=1.0, zorder=7))
    ax.text(ego_x, cy + 0.03, "ego", ha="center", va="center", fontsize=4.8, color=C["green"], fontweight="bold", zorder=8)
    neighbors = [(cx + 0.04, cy + 0.30), (cx + 0.36, cy + 0.05), (cx + 0.20, cy - 0.30), (cx - 0.10, cy - 0.18)]
    for i, (x0, y0) in enumerate(neighbors):
        x1, y1 = neighbors[(i + 1) % len(neighbors)]
        ax.plot([x0, x1], [y0, y1], color="#8fb9a1", linewidth=0.8, zorder=4)
    for x, y in neighbors:
        ax.add_patch(Circle((x, y), 0.075, facecolor=C["green_soft"], edgecolor=C["green"], linewidth=0.7, zorder=6))
    ax.plot([ego_x + 0.13, cx - 0.18], [cy + 0.03, cy + 0.03], color=C["green"], linewidth=0.85, linestyle=(0, (2, 2)), zorder=5)
    ax.add_patch(Circle((cx - 0.39, cy + 0.31), 0.13, facecolor=C["white"], edgecolor=C["green"], linewidth=0.9, zorder=7))
    ax.text(cx - 0.39, cy + 0.31, r"$-I$", ha="center", va="center", fontsize=6.8, color=C["green"], fontweight="bold", zorder=8)


def filter_icon(ax: plt.Axes, cx: float, cy: float, *, color: str, no_self: bool) -> None:
    rounded(ax, cx - 0.58, cy - 0.34, 1.16, 0.68, face=C["white"], edge=color, lw=0.78, radius=0.035, z=3)
    ax.plot([cx - 0.42, cx - 0.42], [cy - 0.14, cy + 0.18], color=C["line"], lw=0.6, zorder=4)
    ax.plot([cx - 0.42, cx + 0.42], [cy - 0.14, cy - 0.14], color=C["line"], lw=0.6, zorder=4)
    if no_self:
        ax.plot([cx - 0.37, cx - 0.06, cx + 0.14, cx + 0.38], [cy - 0.02, cy + 0.08, cy + 0.17, cy - 0.01], color=C["green"], lw=1.35, zorder=6)
        ax.plot([cx - 0.22, cx - 0.05], [cy + 0.21, cy + 0.21], color=C["green"], lw=1.4, zorder=6)
        ax.plot([cx + 0.09, cx + 0.26], [cy + 0.21, cy + 0.21], color=C["red"], lw=1.4, zorder=6)
    else:
        ax.plot([cx - 0.36, cx - 0.12], [cy + 0.13, cy + 0.02], color=C["blue"], lw=1.4, zorder=6)
        ax.plot([cx + 0.08, cx + 0.37], [cy - 0.01, cy + 0.17], color=C["red"], lw=1.4, zorder=6)


def gate_icon(ax: plt.Axes, cx: float, cy: float) -> None:
    ax.add_patch(Circle((cx - 0.16, cy + 0.04), 0.28, facecolor=C["blue_soft"], edgecolor=C["blue"], linewidth=0.9, zorder=4))
    ax.add_patch(Circle((cx + 0.16, cy + 0.04), 0.28, facecolor=C["red_soft"], edgecolor=C["red"], linewidth=0.9, alpha=0.95, zorder=4))
    ax.add_patch(Circle((cx, cy + 0.04), 0.09, facecolor=C["white"], edgecolor=C["blue"], linewidth=0.9, zorder=6))
    ax.plot([cx, cx + 0.36], [cy + 0.04, cy + 0.25], color=C["blue"], linewidth=1.15, zorder=7)


def residual_fusion_icon(ax: plt.Axes, cx: float, cy: float) -> None:
    ax.add_patch(Rectangle((cx - 0.43, cy - 0.23), 0.42, 0.46, facecolor=C["white"], edgecolor=C["green"], linewidth=0.85, zorder=4))
    ax.add_patch(Rectangle((cx + 0.01, cy - 0.23), 0.42, 0.46, facecolor=C["green_soft"], edgecolor=C["green"], linewidth=0.85, zorder=4))
    ax.text(cx - 0.22, cy, r"$B_0$", ha="center", va="center", fontsize=8.2, color=C["ink"], zorder=6)
    ax.text(cx + 0.22, cy, r"$U$", ha="center", va="center", fontsize=8.2, color=C["ink"], zorder=6)
    ax.plot([cx - 0.49, cx - 0.56, cx - 0.56, cx - 0.49], [cy + 0.29, cy + 0.29, cy - 0.29, cy - 0.29], color=C["green"], linewidth=1.0, zorder=6)
    ax.plot([cx + 0.49, cx + 0.56, cx + 0.56, cx + 0.49], [cy + 0.29, cy + 0.29, cy - 0.29, cy - 0.29], color=C["green"], linewidth=1.0, zorder=6)


def score_card(ax: plt.Axes, cx: float, cy: float, *, name: str, score: str, color: str) -> None:
    rounded(ax, cx - 0.51, cy - 0.36, 1.02, 0.72, face=C["white"], edge=color, lw=0.82, radius=0.035, z=3)
    ax.text(cx, cy + 0.19, name, ha="center", va="center", fontsize=6.4, fontweight="bold", color=color, zorder=6)
    ax.plot([cx - 0.27, cx + 0.27], [cy - 0.03, cy - 0.03], color="#d7dde3", linewidth=2.6, solid_capstyle="round", zorder=5)
    ax.plot([cx - 0.27, cx + 0.08], [cy - 0.03, cy - 0.03], color=color, linewidth=2.6, solid_capstyle="round", zorder=6)
    ax.add_patch(Circle((cx + 0.08, cy - 0.03), 0.052, facecolor=C["white"], edgecolor=color, linewidth=0.8, zorder=7))
    ax.text(cx, cy - 0.23, score + " on validation", ha="center", va="center", fontsize=5.55, color=C["muted"], zorder=6)


def expert_lane(
    ax: plt.Axes,
    *,
    y: float,
    expert: str,
    name: str,
    prior: str,
    color: str,
    soft: str,
    ego_separated: bool,
) -> None:
    x, w, h = 2.42, 6.78, 1.46
    rounded(ax, x, y, w, h, face=soft, edge=color, lw=0.95, radius=0.055, z=1)
    lane_header(ax, 2.72, y + h - 0.22, expert, name, prior, color)

    cy = y + 0.60
    xs = [3.52, 5.15, 6.83, 8.48]
    if ego_separated:
        ego_prior_icon(ax, xs[0], cy)
        filter_icon(ax, xs[1], cy, color=color, no_self=True)
        residual_fusion_icon(ax, xs[2], cy)
        captions = ["ego-separated prior", "no-self residual filters", "explicit ego fusion"]
        score_card(ax, xs[3], cy, name="HARP-ESep", score=r"$a_E$", color=color)
    else:
        self_prior_icon(ax, xs[0], cy)
        filter_icon(ax, xs[1], cy, color=color, no_self=False)
        gate_icon(ax, xs[2], cy)
        captions = ["self-loop prior", "low + residual filters", "node-wise gate"]
        score_card(ax, xs[3], cy, name="HARP-GNN", score=r"$a_H$", color=color)

    for cx, label in zip(xs[:3], captions):
        ax.text(cx, y + 0.12, label, ha="center", va="center", fontsize=5.55, color=C["ink"], zorder=7)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 0.59, cy), (b - 0.60, cy), color=color, lw=1.05, ms=8.2)


def router_card(ax: plt.Axes) -> None:
    x, y, w, h = 9.74, 1.03, 2.92, 3.24
    rounded(ax, x, y, w, h, face=C["gold_soft"], edge=C["gold"], lw=1.0, radius=0.06, z=2)
    ax.text(x + w / 2, y + h - 0.23, "Validation-only selector", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C["ink"], zorder=6)

    # Two aligned score bars make the compared evidence explicit.
    bar_x0, bar_x1 = x + 0.84, x + 2.36
    for yy in (y + 2.52, y + 2.16):
        ax.plot([bar_x0, bar_x1], [yy, yy], color="#ded7bd", linewidth=4.0, solid_capstyle="round", zorder=4)
    ax.text(x + 0.43, y + 2.52, r"$a_H$", ha="center", va="center", fontsize=7.2, color=C["blue"], fontweight="bold", zorder=6)
    ax.text(x + 0.43, y + 2.16, r"$a_E$", ha="center", va="center", fontsize=7.2, color=C["green"], fontweight="bold", zorder=6)
    ax.plot([bar_x0, x + 1.72], [y + 2.52, y + 2.52], color=C["blue"], linewidth=4.0, solid_capstyle="round", zorder=5)
    ax.plot([bar_x0, x + 2.09], [y + 2.16, y + 2.16], color=C["green"], linewidth=4.0, solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((x + 1.72, y + 2.52), 0.058, facecolor=C["white"], edgecolor=C["blue"], linewidth=0.85, zorder=7))
    ax.add_patch(Circle((x + 2.09, y + 2.16), 0.058, facecolor=C["white"], edgecolor=C["green"], linewidth=0.85, zorder=7))

    ax.text(x + w / 2, y + 1.75, r"$\Delta_{\rm val}=a_E-a_H$", ha="center", va="center", fontsize=8.3, color=C["ink"], zorder=6)

    axis_left, axis_right, axis_y = x + 0.48, x + 2.44, y + 1.22
    tau_x = x + 1.46
    ax.add_patch(Rectangle((axis_left, axis_y - 0.11), tau_x - axis_left, 0.22, facecolor=C["blue_soft"], edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((tau_x, axis_y - 0.11), axis_right - tau_x, 0.22, facecolor=C["green_soft"], edgecolor="none", zorder=3))
    ax.plot([axis_left, axis_right], [axis_y, axis_y], color=C["gold"], linewidth=1.15, zorder=5)
    ax.plot([tau_x, tau_x], [axis_y - 0.19, axis_y + 0.21], color=C["gold"], linewidth=0.95, linestyle=(0, (2, 2)), zorder=6)
    ax.text(x + 0.92, axis_y + 0.22, "retain H", ha="center", va="bottom", fontsize=5.8, color=C["blue"], fontweight="bold", zorder=6)
    ax.text(x + 2.03, axis_y + 0.22, "select E", ha="center", va="bottom", fontsize=5.8, color=C["green"], fontweight="bold", zorder=6)
    ax.text(tau_x, axis_y - 0.25, r"$\tau=1.96\,\mathrm{SE}$", ha="center", va="top", fontsize=6.2, color=C["gold"], zorder=6)
    ax.text(
        x + w / 2,
        y + 0.38,
        r"switch only if $\Delta_{\rm val}>\tau$",
        ha="center",
        va="center",
        fontsize=6.4,
        color=C["ink"],
        fontweight="bold",
        zorder=6,
    )


def lock_icon(ax: plt.Axes, cx: float, cy: float) -> None:
    ax.add_patch(Rectangle((cx - 0.18, cy - 0.20), 0.36, 0.32, facecolor=C["white"], edgecolor=C["purple"], linewidth=1.05, zorder=6))
    ax.add_patch(Arc((cx, cy + 0.12), 0.30, 0.34, theta1=0, theta2=180, color=C["purple"], linewidth=1.15, zorder=6))
    ax.add_patch(Circle((cx, cy - 0.04), 0.035, facecolor=C["purple"], edgecolor="none", zorder=7))


def test_card(ax: plt.Axes) -> None:
    x, y, w, h = 13.18, 1.18, 1.53, 2.74
    rounded(ax, x, y, w, h, face=C["purple_soft"], edge=C["purple"], lw=1.0, radius=0.055, z=2)
    ax.text(x + w / 2, y + h - 0.25, "Frozen expert", ha="center", va="center", fontsize=7.2, fontweight="bold", color=C["ink"], zorder=6)

    token_y = y + 1.97
    for xx, label, color, soft in [
        (x + 0.44, "H", C["blue"], C["blue_soft"]),
        (x + 1.09, "E", C["green"], C["green_soft"]),
    ]:
        ax.add_patch(Circle((xx, token_y), 0.15, facecolor=soft, edgecolor=color, linewidth=0.9, zorder=6))
        ax.text(xx, token_y, label, ha="center", va="center", fontsize=6.7, color=color, fontweight="bold", zorder=7)
        arrow(ax, (xx, token_y - 0.14), (x + w / 2, y + 1.49), color=C["purple"], lw=0.75, ms=6.8)
    lock_icon(ax, x + w / 2, y + 1.27)
    arrow(ax, (x + w / 2, y + 1.00), (x + w / 2, y + 0.65), color=C["purple"], lw=0.9, ms=7.0)
    ax.text(x + w / 2, y + 0.39, "test score\n+ oracle regret", ha="center", va="center", fontsize=6.05, color=C["ink"], linespacing=1.0, zorder=6)

    # Test labels cannot feed back into selection.
    back_y = 0.76
    ax.plot([14.60, 12.93], [back_y, back_y], color=C["red"], linewidth=0.9, linestyle=(0, (3, 2)), zorder=5)
    arrow(ax, (14.60, back_y), (12.95, back_y), color=C["red"], lw=0.9, ms=7.2, style=(0, (3, 2)))
    cross_x = 13.62
    ax.plot([cross_x - 0.09, cross_x + 0.09], [back_y - 0.12, back_y + 0.12], color=C["red"], linewidth=1.5, zorder=8)
    ax.plot([cross_x - 0.09, cross_x + 0.09], [back_y + 0.12, back_y - 0.12], color=C["red"], linewidth=1.5, zorder=8)
    ax.text(13.95, 0.47, "test labels never route", ha="center", va="center", fontsize=5.9, color=C["red"], fontweight="bold", zorder=6)


def plot_framework(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.55, 3.02), dpi=420)
    ax.set_xlim(0, 15.10)
    ax.set_ylim(0.24, 5.25)
    ax.axis("off")

    stages = [
        (0.14, 2.06, 1, "data split", r"one graph, fixed masks", C["muted"], C["data_bg"]),
        (2.22, 9.42, 2, "parallel experts", "two structural priors, same budget", C["blue"], C["train_bg"]),
        (9.58, 12.84, 3, "confidence routing", "validation evidence only", C["gold"], C["route_bg"]),
        (13.02, 14.92, 4, "locked test", "one final evaluation", C["purple"], C["test_bg"]),
    ]
    for x0, x1, number, title, subtitle, color, fill in stages:
        ax.add_patch(Rectangle((x0, 0.34), x1 - x0, 4.24, facecolor=fill, edgecolor="#e1e5e9", linewidth=0.55, zorder=0))
        stage_header(ax, x0, x1, number, title, subtitle, color)

    for x in (2.14, 9.50, 12.93):
        ax.plot([x, x], [0.43, 4.48], color=C["line"], linewidth=0.75, linestyle=(0, (3, 3)), zorder=1)

    split_card(ax)

    # Equal-budget training is shown once as a shared contract above both lanes.
    ax.plot([2.66, 8.98], [4.43, 4.43], color="#aeb8c2", linewidth=0.75, zorder=3)
    ax.plot([2.66, 2.66], [4.35, 4.43], color="#aeb8c2", linewidth=0.75, zorder=3)
    ax.plot([8.98, 8.98], [4.35, 4.43], color="#aeb8c2", linewidth=0.75, zorder=3)
    ax.text(5.82, 4.43, "identical labels, masks, and optimization budget", ha="center", va="center", fontsize=6.15, color=C["muted"], bbox={"facecolor": C["train_bg"], "edgecolor": "none", "pad": 1.0}, zorder=5)

    expert_lane(
        ax,
        y=2.75,
        expert="H",
        name="HARP-GNN",
        prior="self-loop residual expert",
        color=C["blue"],
        soft=C["blue_soft"],
        ego_separated=False,
    )
    expert_lane(
        ax,
        y=0.98,
        expert="E",
        name="HARP-ESep",
        prior="ego-separated no-self expert",
        color=C["green"],
        soft=C["green_soft"],
        ego_separated=True,
    )

    # The graph split fans out once; colored paths remain traceable to routing.
    arrow(ax, (1.78, 2.73), (2.43, 3.45), color=C["blue"], lw=1.25, rad=0.17, ms=10.0)
    arrow(ax, (1.78, 2.30), (2.43, 1.67), color=C["green"], lw=1.25, rad=-0.17, ms=10.0)

    router_card(ax)
    arrow(ax, (8.99, 3.35), (9.76, 3.25), color=C["blue"], lw=1.2, rad=-0.08, ms=9.5)
    arrow(ax, (8.99, 1.58), (9.76, 2.56), color=C["green"], lw=1.2, rad=0.14, ms=9.5)
    ax.text(9.35, 3.49, r"$a_H$", ha="center", va="center", fontsize=7.3, color=C["blue"], zorder=7)
    ax.text(9.35, 1.94, r"$a_E$", ha="center", va="center", fontsize=7.3, color=C["green"], zorder=7)

    test_card(ax)
    arrow(ax, (12.64, 2.57), (13.20, 2.57), color=C["purple"], lw=1.3, ms=10.0)
    ax.text(12.93, 2.79, "freeze", ha="center", va="center", fontsize=5.7, color=C["purple"], fontweight="bold", zorder=7)

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
