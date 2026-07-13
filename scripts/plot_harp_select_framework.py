from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
    Wedge,
)


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
    "ink": "#202833",
    "muted": "#66727d",
    "line": "#b8c3cc",
    "line_light": "#d9e0e6",
    "outer": "#f3f7fb",
    "white": "#ffffff",
    "navy": "#153f73",
    "navy_dark": "#0c2d54",
    "blue": "#2b6ba6",
    "blue_soft": "#e7f1fb",
    "green": "#37815a",
    "green_soft": "#e7f3eb",
    "red": "#a84b43",
    "red_soft": "#f8e8e5",
    "gold": "#8c6a16",
    "gold_soft": "#fff2c8",
    "purple": "#694f8d",
    "purple_soft": "#f0eafb",
    "warm": "#faf6ef",
    "shadow": "#d6dde4",
}


GRAPH_POS = [
    (-0.52, 0.05),
    (-0.38, 0.43),
    (-0.02, 0.58),
    (0.36, 0.44),
    (0.55, 0.08),
    (0.40, -0.38),
    (0.00, -0.54),
    (-0.43, -0.35),
    (0.00, 0.02),
]
GRAPH_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 0),
    (0, 8),
    (1, 8),
    (2, 8),
    (3, 8),
    (4, 8),
    (5, 8),
    (6, 8),
    (7, 8),
    (1, 7),
    (3, 5),
]
CLASS_FILLS = [
    "#dbeafe",
    "#fee2e2",
    "#dcfce7",
    "#ede9fe",
    "#fef3c7",
    "#dbeafe",
    "#fee2e2",
    "#dcfce7",
    "#ede9fe",
]


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
    edge: str = "#a9b6c1",
    subtitle: str | None = None,
    dashed: bool = False,
    title_size: float = 7.1,
) -> None:
    rounded(
        ax,
        x + 0.045,
        y - 0.045,
        w,
        h,
        face=C["shadow"],
        edge="none",
        lw=0,
        radius=0.11,
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
        lw=0.82,
        radius=0.11,
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
        zorder=12,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h - 0.47,
            subtitle,
            ha="center",
            va="center",
            fontsize=5.45,
            color=C["muted"],
            zorder=12,
        )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = C["navy_dark"],
    lw: float = 0.95,
    rad: float = 0.0,
    ms: float = 8.0,
    style: object = "-",
    z: int = 11,
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


def blend(color: str, amount: float) -> str:
    base = to_rgb(color)
    amount = max(0.0, min(1.0, amount))
    return to_hex(tuple(1.0 - (1.0 - channel) * amount for channel in base))


def signal_color(value: float) -> str:
    if value >= 0:
        return blend(C["red"], 0.22 + 0.72 * min(1.0, value))
    return blend(C["blue"], 0.22 + 0.72 * min(1.0, -value))


def graph_points(cx: float, cy: float, scale: float) -> list[tuple[float, float]]:
    return [(cx + x * scale, cy + y * scale) for x, y in GRAPH_POS]


def draw_graph(
    ax: plt.Axes,
    cx: float,
    cy: float,
    *,
    scale: float,
    edge_color: str = "#758592",
    node_fills: list[str] | None = None,
    outline: str | None = None,
    loops: bool = False,
    halo: int | None = None,
    edge_alpha: float = 0.72,
    lw: float = 0.72,
) -> None:
    points = graph_points(cx, cy, scale)
    for i, j in GRAPH_EDGES:
        ax.plot(
            [points[i][0], points[j][0]],
            [points[i][1], points[j][1]],
            color=edge_color,
            linewidth=lw,
            alpha=edge_alpha,
            zorder=4,
        )
    if halo is not None:
        hx, hy = points[halo]
        ax.add_patch(
            Circle(
                (hx, hy),
                0.17 * scale,
                facecolor=blend(C["green"], 0.18),
                edgecolor=C["green"],
                linewidth=0.7,
                alpha=0.55,
                zorder=4,
            )
        )
    fills = node_fills or CLASS_FILLS
    stroke = outline or edge_color
    for idx, ((x, y), fill) in enumerate(zip(points, fills)):
        radius = (0.072 if idx != 8 else 0.082) * scale
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=fill,
                edgecolor=stroke,
                linewidth=0.58,
                zorder=6,
            )
        )
        if loops and idx in (2, 6, 8):
            ax.add_patch(
                Arc(
                    (x, y),
                    0.22 * scale,
                    0.19 * scale,
                    theta1=20,
                    theta2=325,
                    color=stroke,
                    linewidth=0.62,
                    zorder=5,
                )
            )


def draw_signal_graph(
    ax: plt.Axes,
    cx: float,
    cy: float,
    *,
    scale: float,
    values: list[float],
    outline: str = "#687681",
) -> None:
    fills = [signal_color(value) for value in values]
    draw_graph(
        ax,
        cx,
        cy,
        scale=scale,
        edge_color="#9aa6af",
        node_fills=fills,
        outline=outline,
        edge_alpha=0.62,
        lw=0.62,
    )


def draw_ego_view(ax: plt.Axes, cx: float, cy: float, *, scale: float) -> None:
    ego = (cx - 0.43 * scale, cy)
    ring = [
        (cx + 0.03 * scale, cy + 0.36 * scale),
        (cx + 0.43 * scale, cy + 0.14 * scale),
        (cx + 0.35 * scale, cy - 0.31 * scale),
        (cx - 0.02 * scale, cy - 0.40 * scale),
    ]
    for i, start in enumerate(ring):
        end = ring[(i + 1) % len(ring)]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#8caf99",
            linewidth=0.72,
            zorder=4,
        )
    for x, y in ring:
        ax.add_patch(
            Circle(
                (x, y),
                0.075 * scale,
                facecolor=C["green_soft"],
                edgecolor=C["green"],
                linewidth=0.62,
                zorder=6,
            )
        )
    ax.add_patch(
        Circle(
            ego,
            0.13 * scale,
            facecolor=C["white"],
            edgecolor=C["green"],
            linewidth=0.9,
            zorder=7,
        )
    )
    ax.text(
        ego[0],
        ego[1],
        r"$i$",
        ha="center",
        va="center",
        fontsize=max(5.2, 7.0 * scale),
        color=C["green"],
        fontweight="bold",
        zorder=8,
    )
    ax.plot(
        [ego[0] + 0.15 * scale, cx - 0.14 * scale],
        [cy, cy],
        color=C["green"],
        linewidth=0.72,
        linestyle=(0, (2, 2)),
        zorder=5,
    )


def draw_split_bar(ax: plt.Axes, cx: float, y: float, *, width: float, labels: bool = True) -> None:
    left = cx - width / 2
    segments = [(0.58, C["blue"]), (0.22, C["gold"]), (0.20, C["purple"])]
    cursor = left
    for fraction, color in segments:
        segment_width = width * fraction
        ax.add_patch(
            Rectangle(
                (cursor, y),
                segment_width,
                0.13,
                facecolor=color,
                edgecolor=C["white"],
                linewidth=0.35,
                zorder=7,
            )
        )
        cursor += segment_width
    if labels:
        ax.text(left + width * 0.29, y - 0.11, "train", ha="center", va="center", fontsize=5.1, color=C["muted"], zorder=8)
        ax.text(left + width * 0.69, y - 0.11, "val", ha="center", va="center", fontsize=5.1, color=C["muted"], zorder=8)
        ax.text(left + width * 0.90, y - 0.11, "test", ha="center", va="center", fontsize=5.1, color=C["muted"], zorder=8)


def draw_basis_bank(
    ax: plt.Axes,
    cx: float,
    cy: float,
    *,
    color: str,
    scale: float,
    no_self: bool,
) -> None:
    w, h = 1.05 * scale, 0.62 * scale
    rounded(ax, cx - w / 2, cy - h / 2, w, h, face=C["white"], edge=color, lw=0.72, radius=0.035, z=3)
    x0, y0 = cx - 0.39 * scale, cy - 0.19 * scale
    x1, y1 = cx + 0.40 * scale, cy + 0.22 * scale
    ax.plot([x0, x0], [y0, y1], color=C["line"], lw=0.52, zorder=4)
    ax.plot([x0, x1], [y0, y0], color=C["line"], lw=0.52, zorder=4)
    xs = [x0 + (x1 - x0) * i / 18 for i in range(19)]
    low = []
    high = []
    residual = []
    for i in range(19):
        t = i / 18
        if no_self:
            low.append(cy - 0.07 * scale + 0.20 * scale * math.sin(0.9 * math.pi * t))
            high.append(cy + 0.18 * scale - 0.35 * scale * abs(t - 0.54))
        else:
            low.append(cy + 0.17 * scale - 0.27 * scale * t)
            high.append(cy - 0.10 * scale + 0.28 * scale * t)
        residual.append(cy - 0.10 * scale + 0.23 * scale * math.sin(math.pi * t))
    ax.plot(xs, low, color=color, lw=1.1, zorder=6)
    ax.plot(xs, high, color=C["red"], lw=1.0, zorder=6)
    ax.plot(xs, residual, color=C["gold"], lw=0.72, linestyle=(0, (2.4, 1.8)), zorder=6)


def draw_node_gate(ax: plt.Axes, cx: float, cy: float, *, scale: float) -> None:
    offsets = [(-0.23, 0.15), (0.05, 0.25), (0.26, 0.02), (0.11, -0.24), (-0.20, -0.17)]
    weights = [0.18, 0.42, 0.82, 0.63, 0.30]
    for (dx, dy), weight in zip(offsets, weights):
        x, y = cx + dx * scale, cy + dy * scale
        radius = 0.105 * scale
        ax.add_patch(Wedge((x, y), radius, 90, 90 + 360 * weight, facecolor=C["red_soft"], edgecolor="none", zorder=5))
        ax.add_patch(Wedge((x, y), radius, 90 + 360 * weight, 450, facecolor=C["blue_soft"], edgecolor="none", zorder=5))
        ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor=C["navy"], linewidth=0.55, zorder=6))
    ax.plot([cx - 0.23 * scale, cx + 0.05 * scale, cx + 0.26 * scale, cx + 0.11 * scale, cx - 0.20 * scale, cx - 0.23 * scale],
            [cy + 0.15 * scale, cy + 0.25 * scale, cy + 0.02 * scale, cy - 0.24 * scale, cy - 0.17 * scale, cy + 0.15 * scale],
            color=C["line"], lw=0.50, zorder=4)
    ax.text(cx, cy - 0.40 * scale, r"$g_i$", ha="center", va="center", fontsize=6.0, color=C["ink"], zorder=8)


def draw_feature_fusion(ax: plt.Axes, cx: float, cy: float, *, scale: float) -> None:
    cell_w, cell_h = 0.28 * scale, 0.43 * scale
    left = cx - cell_w
    ax.add_patch(Rectangle((left, cy - cell_h / 2), cell_w, cell_h, facecolor=C["white"], edgecolor=C["green"], linewidth=0.75, zorder=5))
    ax.add_patch(Rectangle((cx, cy - cell_h / 2), cell_w, cell_h, facecolor=C["green_soft"], edgecolor=C["green"], linewidth=0.75, zorder=5))
    ax.text(left + cell_w / 2, cy, r"$B_0$", ha="center", va="center", fontsize=7.2 * scale, color=C["ink"], zorder=7)
    ax.text(cx + cell_w / 2, cy, r"$U_i$", ha="center", va="center", fontsize=7.2 * scale, color=C["ink"], zorder=7)
    ax.plot([left - 0.10 * scale, left - 0.16 * scale, left - 0.16 * scale, left - 0.10 * scale],
            [cy + cell_h / 2 + 0.05 * scale, cy + cell_h / 2 + 0.05 * scale, cy - cell_h / 2 - 0.05 * scale, cy - cell_h / 2 - 0.05 * scale],
            color=C["green"], lw=0.85, zorder=7)
    right = cx + cell_w
    ax.plot([right + 0.10 * scale, right + 0.16 * scale, right + 0.16 * scale, right + 0.10 * scale],
            [cy + cell_h / 2 + 0.05 * scale, cy + cell_h / 2 + 0.05 * scale, cy - cell_h / 2 - 0.05 * scale, cy - cell_h / 2 - 0.05 * scale],
            color=C["green"], lw=0.85, zorder=7)


def draw_score_interval(ax: plt.Axes, cx: float, cy: float, *, color: str, label: str, scale: float = 1.0) -> None:
    rounded(ax, cx - 0.48 * scale, cy - 0.25 * scale, 0.96 * scale, 0.50 * scale, face=C["white"], edge=color, lw=0.70, radius=0.035, z=4)
    ax.text(cx - 0.27 * scale, cy + 0.10 * scale, label, ha="center", va="center", fontsize=6.6 * scale, color=color, fontweight="bold", zorder=7)
    ax.plot([cx - 0.07 * scale, cx + 0.29 * scale], [cy + 0.10 * scale, cy + 0.10 * scale], color=C["line_light"], lw=2.2 * scale, solid_capstyle="round", zorder=5)
    ax.plot([cx - 0.07 * scale, cx + 0.16 * scale], [cy + 0.10 * scale, cy + 0.10 * scale], color=color, lw=2.2 * scale, solid_capstyle="round", zorder=6)
    ax.add_patch(Circle((cx + 0.16 * scale, cy + 0.10 * scale), 0.035 * scale, facecolor=C["white"], edgecolor=color, linewidth=0.65, zorder=7))
    ax.text(cx, cy - 0.12 * scale, "validation", ha="center", va="center", fontsize=5.2 * scale, color=C["muted"], zorder=7)


def draw_matrix(ax: plt.Axes, cx: float, cy: float, *, color: str, scale: float, cols: int = 3, rows: int = 3) -> None:
    cell = 0.13 * scale
    left = cx - cols * cell / 2
    bottom = cy - rows * cell / 2
    levels = [0.25, 0.55, 0.82, 0.42, 0.70, 0.32, 0.62, 0.88, 0.48]
    for row in range(rows):
        for col in range(cols):
            idx = (row * cols + col) % len(levels)
            ax.add_patch(
                Rectangle(
                    (left + col * cell, bottom + row * cell),
                    cell,
                    cell,
                    facecolor=blend(color, levels[idx]),
                    edgecolor=C["white"],
                    linewidth=0.35,
                    zorder=6,
                )
            )
    ax.add_patch(Rectangle((left, bottom), cols * cell, rows * cell, facecolor="none", edgecolor=color, linewidth=0.7, zorder=7))


def draw_lock(ax: plt.Axes, cx: float, cy: float, *, scale: float) -> None:
    ax.add_patch(Rectangle((cx - 0.16 * scale, cy - 0.17 * scale), 0.32 * scale, 0.27 * scale, facecolor=C["white"], edgecolor=C["purple"], linewidth=0.95, zorder=8))
    ax.add_patch(Arc((cx, cy + 0.10 * scale), 0.27 * scale, 0.30 * scale, theta1=0, theta2=180, color=C["purple"], linewidth=1.0, zorder=8))
    ax.add_patch(Circle((cx, cy - 0.03 * scale), 0.027 * scale, facecolor=C["purple"], edgecolor="none", zorder=9))


def draw_input_protocol(ax: plt.Axes) -> None:
    panel(
        ax,
        0.42,
        3.87,
        2.45,
        2.10,
        title="Fixed Graph Split",
        subtitle="fixed masks for both experts",
        face=C["white"],
        title_size=7.15,
    )
    draw_graph(ax, 1.64, 4.78, scale=0.88, edge_color="#748490")
    draw_split_bar(ax, 1.64, 4.09, width=1.58)

    panel(
        ax,
        0.42,
        0.48,
        2.45,
        3.12,
        title="Structural Views",
        subtitle="same nodes and features",
        face="#f9fbfd",
        dashed=True,
        title_size=7.05,
    )
    ax.add_patch(Circle((1.64, 2.88), 0.16, facecolor=C["white"], edgecolor=C["navy"], linewidth=0.8, zorder=7))
    ax.text(1.64, 2.88, r"$G$", ha="center", va="center", fontsize=7.0, color=C["navy"], fontweight="bold", zorder=8)
    arrow(ax, (1.52, 2.74), (1.12, 2.45), color=C["blue"], lw=0.78, ms=6.8)
    arrow(ax, (1.76, 2.74), (2.18, 2.45), color=C["green"], lw=0.78, ms=6.8)
    draw_graph(
        ax,
        1.10,
        1.88,
        scale=0.67,
        edge_color=C["blue"],
        node_fills=[C["white"]] * len(GRAPH_POS),
        outline=C["blue"],
        loops=True,
        edge_alpha=0.75,
    )
    draw_ego_view(ax, 2.18, 1.88, scale=0.72)
    ax.text(1.10, 1.27, r"self-loop $\hat A$", ha="center", va="center", fontsize=5.85, color=C["blue"], fontweight="bold", zorder=8)
    ax.text(2.18, 1.27, r"no-self $\tilde A$", ha="center", va="center", fontsize=5.85, color=C["green"], fontweight="bold", zorder=8)
    rounded(ax, 0.65, 0.72, 0.93, 0.31, face=C["blue"], edge=C["navy_dark"], lw=0.45, radius=0.05, z=7)
    rounded(ax, 1.70, 0.72, 0.93, 0.31, face=C["green"], edge="#245d40", lw=0.45, radius=0.05, z=7)
    ax.text(1.115, 0.875, "H: retain", ha="center", va="center", fontsize=5.45, color=C["white"], fontweight="bold", zorder=8)
    ax.text(2.165, 0.875, "E: separate", ha="center", va="center", fontsize=5.45, color=C["white"], fontweight="bold", zorder=8)


def draw_expert_lanes(ax: plt.Axes) -> None:
    panel(
        ax,
        3.18,
        3.33,
        8.30,
        2.64,
        title="Parallel residual graph experts",
        subtitle="identical labels, masks, optimizer, and training budget",
        face=C["warm"],
        title_size=7.55,
    )

    column_x = [4.72, 6.08, 7.47, 8.94, 10.58]
    column_labels = ["STRUCTURE", "FILTER BANK", "NODE FIELD", "FUSION", "VAL. SCORE"]
    for x, label in zip(column_x, column_labels):
        ax.text(x, 5.30, label, ha="center", va="center", fontsize=5.0, color=C["muted"], fontweight="bold", zorder=10)

    rounded(ax, 3.38, 4.31, 7.90, 0.88, face=C["blue_soft"], edge=C["blue"], lw=0.82, radius=0.06, z=2)
    rounded(ax, 3.38, 3.43, 7.90, 0.80, face=C["green_soft"], edge=C["green"], lw=0.82, radius=0.06, z=2)

    for cy, token, name, color in [
        (4.75, "H", "HARP-\nGNN", C["blue"]),
        (3.83, "E", "HARP-\nESep", C["green"]),
    ]:
        ax.add_patch(Circle((3.62, cy), 0.115, facecolor=C["white"], edgecolor=color, linewidth=0.85, zorder=8))
        ax.text(3.62, cy, token, ha="center", va="center", fontsize=5.8, color=color, fontweight="bold", zorder=9)
        ax.text(4.08, cy, name, ha="center", va="center", fontsize=5.05, color=color, fontweight="bold", linespacing=0.83, zorder=9)

    top_y, bottom_y = 4.70, 3.82
    draw_graph(ax, column_x[0], top_y, scale=0.40, edge_color=C["blue"], node_fills=[C["white"]] * len(GRAPH_POS), outline=C["blue"], loops=True)
    draw_basis_bank(ax, column_x[1], top_y, color=C["blue"], scale=0.82, no_self=False)
    draw_signal_graph(ax, column_x[2], top_y, scale=0.39, values=[-0.8, 0.2, 0.8, -0.4, 0.6, -0.7, 0.3, 0.9, -0.1], outline=C["blue"])
    draw_node_gate(ax, column_x[3], top_y, scale=0.92)
    draw_score_interval(ax, column_x[4], top_y, color=C["blue"], label=r"$a_H$", scale=0.90)

    draw_ego_view(ax, column_x[0], bottom_y, scale=0.55)
    draw_basis_bank(ax, column_x[1], bottom_y, color=C["green"], scale=0.82, no_self=True)
    draw_signal_graph(ax, column_x[2], bottom_y, scale=0.37, values=[0.7, -0.6, 0.1, 0.8, -0.3, 0.5, -0.9, 0.4, 0.2], outline=C["green"])
    draw_feature_fusion(ax, column_x[3], bottom_y, scale=0.90)
    draw_score_interval(ax, column_x[4], bottom_y, color=C["green"], label=r"$a_E$", scale=0.90)

    for y, color in [(top_y, C["blue"]), (bottom_y, C["green"])]:
        for start, end in [(5.12, 5.57), (6.57, 7.00), (7.91, 8.45), (9.42, 10.07)]:
            arrow(ax, (start, y), (end, y), color=color, lw=0.82, ms=6.8)


def draw_mechanism_zoom(ax: plt.Axes) -> None:
    panel(
        ax,
        3.18,
        0.48,
        8.30,
        2.57,
        title="Mechanism zoom: complementary residual bases",
        face="#fbfcfd",
        dashed=True,
        title_size=7.35,
    )
    divider_x = 7.34
    ax.plot([divider_x, divider_x], [0.77, 2.58], color=C["line_light"], lw=0.72, linestyle=(0, (2, 2)), zorder=3)

    ax.text(5.25, 2.52, "H: residual node fields", ha="center", va="center", fontsize=5.9, color=C["blue"], fontweight="bold", zorder=8)
    snapshot_x = [4.08, 5.25, 6.42]
    values = [
        [-0.2, 0.7, 0.4, -0.8, 0.3, -0.5, 0.8, -0.1, 0.6],
        [-0.1, 0.4, 0.2, -0.5, 0.2, -0.3, 0.5, 0.0, 0.4],
        [-0.1, 0.3, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2],
    ]
    for x, node_values in zip(snapshot_x, values):
        draw_signal_graph(ax, x, 1.66, scale=0.56, values=node_values, outline=C["blue"])
    ax.text(snapshot_x[0], 2.18, r"$L_{k-1}$", ha="center", va="center", fontsize=6.2, color=C["ink"], zorder=8)
    ax.text(snapshot_x[1], 2.18, r"$L_k$", ha="center", va="center", fontsize=6.2, color=C["ink"], zorder=8)
    ax.text(snapshot_x[2], 2.18, r"$H_k$", ha="center", va="center", fontsize=6.2, color=C["ink"], zorder=8)
    ax.text(4.67, 1.66, r"$-$", ha="center", va="center", fontsize=9.5, color=C["navy"], fontweight="bold", zorder=9)
    ax.text(5.84, 1.66, r"$=$", ha="center", va="center", fontsize=8.5, color=C["navy"], fontweight="bold", zorder=9)
    ax.text(5.25, 0.80, r"$L_k=\hat A^kX,\qquad H_k=L_{k-1}-L_k$", ha="center", va="center", fontsize=6.0, color=C["ink"], zorder=8)

    ax.text(9.40, 2.52, "E: no-self propagation and fusion", ha="center", va="center", fontsize=5.9, color=C["green"], fontweight="bold", zorder=8)
    draw_matrix(ax, 7.90, 1.65, color=C["green"], scale=1.0)
    ax.text(7.90, 2.18, r"$B_0$", ha="center", va="center", fontsize=6.2, color=C["ink"], zorder=8)
    draw_ego_view(ax, 9.16, 1.65, scale=0.72)
    ax.text(9.16, 2.18, r"$B_k=\tilde A^kB_0$", ha="center", va="center", fontsize=6.0, color=C["ink"], zorder=8)
    draw_feature_fusion(ax, 10.65, 1.65, scale=1.12)
    ax.text(10.65, 2.18, r"$[B_0,U_i]$", ha="center", va="center", fontsize=6.2, color=C["ink"], zorder=8)
    arrow(ax, (8.25, 1.65), (8.67, 1.65), color=C["green"], lw=0.82, ms=6.8)
    arrow(ax, (9.67, 1.65), (10.11, 1.65), color=C["green"], lw=0.82, ms=6.8)
    ax.text(9.40, 0.80, "explicit ego anchor + filtered neighbors", ha="center", va="center", fontsize=5.45, color=C["muted"], zorder=8)


def draw_validation_evidence(ax: plt.Axes) -> None:
    panel(
        ax,
        11.73,
        3.33,
        3.82,
        2.64,
        title="Validation evidence",
        subtitle="validation labels only",
        face=C["gold_soft"],
        edge=C["gold"],
        title_size=7.35,
    )
    x_left, x_right = 12.42, 15.13
    interval_specs = [
        (5.00, 13.02, 14.16, 13.74, C["blue"], r"$a_H$"),
        (4.69, 13.18, 14.54, 14.18, C["green"], r"$a_E$"),
    ]
    for y, lo, hi, point, color, label in interval_specs:
        ax.text(12.12, y, label, ha="center", va="center", fontsize=6.8, color=color, fontweight="bold", zorder=9)
        ax.plot([x_left, x_right], [y, y], color="#e2d9ba", lw=1.0, zorder=4)
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round", zorder=6)
        ax.plot([lo, lo], [y - 0.07, y + 0.07], color=color, lw=0.8, zorder=6)
        ax.plot([hi, hi], [y - 0.07, y + 0.07], color=color, lw=0.8, zorder=6)
        ax.add_patch(Circle((point, y), 0.055, facecolor=C["white"], edgecolor=color, linewidth=0.8, zorder=8))

    base_y = 3.83
    curve_left, curve_right = 12.20, 15.12
    center, sigma, amp = 13.84, 0.40, 0.44
    xs = [curve_left + (curve_right - curve_left) * i / 70 for i in range(71)]
    ys = [base_y + amp * math.exp(-0.5 * ((x - center) / sigma) ** 2) for x in xs]
    polygon = [(curve_left, base_y)] + list(zip(xs, ys)) + [(curve_right, base_y)]
    ax.add_patch(Polygon(polygon, closed=True, facecolor=blend(C["gold"], 0.22), edgecolor=C["gold"], linewidth=0.78, zorder=4))
    tau_x = 13.22
    ax.plot([tau_x, tau_x], [base_y - 0.08, base_y + 0.53], color=C["red"], lw=0.85, linestyle=(0, (2.3, 2.0)), zorder=7)
    ax.plot([curve_left, curve_right], [base_y, base_y], color=C["gold"], lw=0.78, zorder=6)
    ax.text(tau_x, base_y - 0.16, r"$\tau$", ha="center", va="center", fontsize=6.2, color=C["red"], zorder=8)
    ax.text(center, base_y + 0.23, r"$\Delta_{\rm val}$", ha="center", va="center", fontsize=6.4, color=C["ink"], zorder=8)
    ax.text(13.64, 3.51, r"select E only if $\Delta_{\rm val}>1.96\,\mathrm{SE}$", ha="center", va="center", fontsize=5.55, color=C["ink"], zorder=8)


def draw_locked_test(ax: plt.Axes) -> None:
    panel(
        ax,
        11.73,
        0.48,
        3.82,
        2.57,
        title="Frozen choice and test",
        subtitle="route first, evaluate second",
        face=C["purple_soft"],
        edge=C["purple"],
        title_size=7.25,
    )
    for x, token, color, soft in [
        (12.45, "H", C["blue"], C["blue_soft"]),
        (14.82, "E", C["green"], C["green_soft"]),
    ]:
        ax.add_patch(Circle((x, 2.25), 0.17, facecolor=soft, edgecolor=color, linewidth=0.9, zorder=7))
        ax.text(x, 2.25, token, ha="center", va="center", fontsize=6.7, color=color, fontweight="bold", zorder=8)
        arrow(ax, (x, 2.10), (13.63, 1.91), color=C["purple"], lw=0.75, ms=6.5)

    ax.add_patch(Polygon([(13.63, 2.05), (13.40, 1.82), (13.86, 1.82)], closed=True, facecolor=C["white"], edgecolor=C["purple"], linewidth=0.85, zorder=7))
    ax.text(13.63, 1.90, r"$e^*$", ha="center", va="center", fontsize=6.2, color=C["purple"], fontweight="bold", zorder=8)
    arrow(ax, (13.63, 1.80), (13.63, 1.58), color=C["purple"], lw=0.8, ms=6.5)

    draw_signal_graph(ax, 13.63, 1.30, scale=0.50, values=[0.4, -0.6, 0.8, -0.2, 0.6, -0.7, 0.3, 0.9, -0.1], outline=C["purple"])
    draw_lock(ax, 13.63, 1.30, scale=0.82)

    draw_split_bar(ax, 13.63, 0.74, width=1.60, labels=False)
    ax.text(13.63, 0.57, "test labels never route", ha="center", va="center", fontsize=5.45, color=C["red"], fontweight="bold", zorder=9)
    ax.plot([12.23, 15.03], [0.91, 0.91], color=C["red"], lw=0.72, linestyle=(0, (3, 2)), zorder=5)
    ax.plot([13.52, 13.74], [0.82, 1.00], color=C["red"], lw=1.15, zorder=8)
    ax.plot([13.52, 13.74], [1.00, 0.82], color=C["red"], lw=1.15, zorder=8)


def plot_framework(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.35, 3.30), dpi=450)
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.14, 7.15)
    ax.axis("off")

    rounded(ax, 0.12, 0.20, 15.76, 6.78, face=C["outer"], edge="#99a9b6", lw=0.86, radius=0.15, z=0)
    ax.text(0.55, 6.66, "HARP-Select:", ha="left", va="center", fontsize=9.7, fontweight="bold", color=C["navy_dark"], zorder=15)
    ax.text(
        3.11,
        6.66,
        "Validation-Calibrated Selection of Complementary Residual Graph Experts",
        ha="left",
        va="center",
        fontsize=7.45,
        fontweight="bold",
        color=C["ink"],
        zorder=15,
    )

    draw_input_protocol(ax)
    draw_expert_lanes(ax)
    draw_mechanism_zoom(ax)
    draw_validation_evidence(ax)
    draw_locked_test(ax)

    arrow(ax, (2.87, 5.25), (3.19, 5.25), color=C["navy_dark"], lw=1.0, ms=8.5)
    arrow(ax, (2.82, 2.35), (3.40, 4.70), color=C["blue"], lw=0.88, rad=-0.18, ms=7.5)
    arrow(ax, (2.82, 1.50), (3.40, 3.82), color=C["green"], lw=0.88, rad=-0.17, ms=7.5)

    arrow(ax, (11.28, 4.70), (11.75, 4.95), color=C["blue"], lw=0.95, rad=-0.08, ms=8.0)
    arrow(ax, (11.28, 3.82), (11.75, 4.65), color=C["green"], lw=0.95, rad=-0.12, ms=8.0)
    arrow(ax, (13.64, 3.33), (13.64, 3.06), color=C["purple"], lw=1.0, ms=8.0)
    arrow(ax, (7.33, 3.33), (7.33, 3.06), color=C["navy"], lw=0.75, ms=6.5, style=(0, (3, 2)))

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
