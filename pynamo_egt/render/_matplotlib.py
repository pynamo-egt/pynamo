"""Matplotlib renderer for scenes."""

from __future__ import annotations

import warnings
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..drawer import DEFAULT_PLOT_STYLE, PlottingWarning, _draw_arrow_2d

_STYLE_DEFAULTS: dict[str, Any] = {
    **{
        key: DEFAULT_PLOT_STYLE[key]
        for key in (
            "figsize", "simplex_font_size", "simplex_zorder", "trajectory_color",
            "trajectory_linewidth", "arrow_size", "arrow_width", "trajectory_zorder",
            "show_trajectories", "show_equilibria", "sink_color", "saddle_color",
            "source_color", "center_color", "equilibrium_size", "equilibrium_edgecolor",
            "equilibrium_zorder",
        )
    },
    "title_pad": 18,
}
# Offsets of vertex labels from their vertex, in plot units.
_LABEL_OFFSETS = {"above": 0.04, "below": -0.08}


def render(
    scene: dict[str, Any], ax: Optional[Axes] = None, fig: Optional[Figure] = None, **style: Any
) -> tuple[Figure, Axes]:
    """Draw a triangle scene; see :func:`pynamo_egt.render.matplotlib`."""
    unknown = set(style) - set(_STYLE_DEFAULTS)
    if unknown:
        raise TypeError(f"Unknown style options: {', '.join(sorted(unknown))}.")
    s = {**_STYLE_DEFAULTS, **style}
    if scene["state_space"]["kind"] != "triangle":
        raise NotImplementedError(f"Cannot draw a {scene['state_space']['kind']} yet.")

    if ax is None:
        fig = plt.figure(figsize=s["figsize"]) if fig is None else fig
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
    figure = cast(Figure, ax.figure)

    _draw_frame(ax, scene["frame"], s)
    if s["show_equilibria"]:
        _draw_rest_points(ax, scene["rest_points"], s)
    if s["show_trajectories"]:
        colors = _trajectory_colors(s["trajectory_color"], len(scene["trajectories"]))
        for trajectory, color in zip(scene["trajectories"], colors):
            for direction in ("forward", "backward"):
                path = np.asarray(trajectory[direction])
                _line(ax, path, color, s)
            for tail, head in trajectory["arrows"]:
                _arrow(figure, ax, tail, head, color, s)
        edge_color = colors[0] if colors else s["trajectory_color"]
        for edge in scene["edge_flow"]:
            _line(ax, np.asarray(edge["segment"]), edge_color, s)
            for tail, head in edge["arrows"]:
                _arrow(figure, ax, tail, head, edge_color, s)
    if scene["title"]:
        ax.set_title(scene["title"], pad=s["title_pad"])
    return figure, ax


def _draw_frame(ax: Axes, frame: dict[str, Any], s: dict[str, Any]) -> None:
    for (x0, y0), (x1, y1) in frame["edges"]:
        ax.plot([x0, x1], [y0, y1], color="black", zorder=s["simplex_zorder"], alpha=1, clip_on=False)
    for label in frame["vertex_labels"]:
        x, y = label["position"]
        ax.annotate(label["text"], (x, y + _LABEL_OFFSETS[label["side"]]), ha="center",
                    fontsize=s["simplex_font_size"], zorder=s["simplex_zorder"])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.10, (3 ** 0.5) / 2 + 0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def _draw_rest_points(ax: Axes, rest_points: list[dict[str, Any]], s: dict[str, Any]) -> None:
    groups: dict[str, list] = {"source": [], "saddle": [], "sink": [], "center": [], "undetermined": []}
    for point in rest_points:
        stability = "source" if point["stability"] == "unstable" else point["stability"]
        groups.get(stability, groups["undetermined"]).append(point["plot"])
    if any(point["stability"] == "unstable" for point in rest_points):
        warnings.warn(
            "At least one equilibrium is classified as 'unstable': linearization "
            "proves it is not stable, but does not distinguish source from saddle. "
            "It is drawn with the source color.",
            PlottingWarning,
            stacklevel=4,
        )
    edge = s["equilibrium_edgecolor"] if s["equilibrium_edgecolor"] is not None else "none"
    center_color = s["source_color"] if s["center_color"] is None else s["center_color"]
    face = {"source": s["source_color"], "saddle": s["saddle_color"], "sink": s["sink_color"],
            "center": center_color, "undetermined": "none"}
    for stability, points in groups.items():
        if not points:
            continue
        pts = np.asarray(points)
        ax.scatter(pts[:, 0], pts[:, 1], s=s["equilibrium_size"], facecolors=face[stability],
                   edgecolors=edge, alpha=1, zorder=s["equilibrium_zorder"], clip_on=False)


def _line(ax: Axes, path: np.ndarray, color, s: dict[str, Any]) -> None:
    ax.plot(path[:, 0], path[:, 1], linewidth=s["trajectory_linewidth"], color=color,
            zorder=s["trajectory_zorder"], clip_on=False)


def _arrow(fig: Figure, ax: Axes, tail, head, color, s: dict[str, Any]) -> None:
    _draw_arrow_2d(np.asarray(tail), np.asarray(head), fig, ax, arrow_size=s["arrow_size"],
                   arrow_width=s["arrow_width"], arrow_color=color, zorder=s["trajectory_zorder"])


def _trajectory_colors(color, count: int) -> list:
    if mcolors.is_color_like(color):
        return [color] * count
    if len(color) != count:
        raise ValueError("trajectory_color must be one color or one color per trajectory.")
    return list(color)
