"""Plotting utilities for pyNamo replicator dynamics.

The main public entry point is phase_portrait. Lower-level functions remain
available for advanced users who want direct control over state spaces,
trajectories, speed fields, vector fields, and equilibrium markers.
"""

import math
import warnings
import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import analysis
from . import dynamics
from .game import infer_game_class, require_supported_game_class

__all__ = [
    "PlottingWarning",
    "simplex_to_plane_2p3s",
    "plane_to_simplex_2p3s",
    "simplex_to_plane_2p4s",
    "plane_to_simplex_2p4s",
    "draw_state_space",
    "plot_trajectory",
    "plot_equilibria",
    "plot_speed_field",
    "plot_vector_field",
    "phase_portrait",
    "bifurcation_diagram",
]


class PlottingWarning(RuntimeWarning):
    """Warning raised when plotting compresses analysis categories."""


DEFAULT_PLOT_STYLE = {
    "figsize": (6, 6),
    "view_elev": 25,
    "view_azim": 35,
    "simplex_font_size": 13,
    "simplex_zorder": 30,
    "show_speed": True,
    "speed_grid": 60,
    "speed_cmap": plt.cm.Spectral_r,
    "speed_levels": 12,
    "speed_zorder": 10,
    "show_vector_field": False,
    "vector_grid": 15,
    "vector_margin": 0.02,
    "vector_color": "black",
    "vector_alpha": 0.75,
    "vector_length": 0.04,
    "vector_width": 0.003,
    "vector_zorder": 15,
    "vector_normalize": True,
    "show_faces": False,
    "face_colors": None,
    "face_alpha": 0.15,
    "face_zorder": 0,
    "show_trajectories": True,
    "trajectory_step": 0.02,
    "trajectory_arrows": [0.02],
    "trajectory_number": 4,
    "show_edge_flow": None,
    "tmax": 45,
    "trajectory_color": "black",
    "trajectory_linewidth": 1.2,
    "arrow_size": 0.04,
    "arrow_width": 0.015,
    "trajectory_zorder": 20,
    "show_equilibria": True,
    "sink_color": "black",
    "saddle_color": "gray",
    "source_color": "white",
    "center_color": None,
    "equilibrium_size": 80,
    "equilibrium_edgecolor": "black",
    "equilibrium_zorder": 40,
}


def simplex_to_plane_2p3s(x, y):
    """Convert simplex coordinates to 1Pop3S plotting plane."""
    return [-0.5 * x - y + 1, (np.sqrt(3) / 2) * x]


def plane_to_simplex_2p3s(x, y):
    """Convert 1Pop3S plotting plane coordinates back to simplex coordinates."""
    return [2 / 3 * np.sqrt(3) * y, -1 / 3 * np.sqrt(3) * y - x + 1]


def simplex_to_plane_2p4s(x, y, z):
    """Convert simplex coordinates to 1Pop4S 3D plotting space."""
    return [0.5 * (-y + z + 1), np.sqrt(3) / 4 * (x - y - z + 1), -np.sqrt(13) / 4 * (x + y + z - 1)]


def plane_to_simplex_2p4s(x, y, z):
    """Convert 1Pop4S plotting space coordinates back to simplex coordinates."""
    return [2 * (np.sqrt(3) / 3 * y - np.sqrt(13) / 13 * z), -x + np.sqrt(3) / 3 * y - np.sqrt(13) / 13 * z + 1, x + np.sqrt(3) / 3 * y - np.sqrt(13) / 13 * z]


def _arrow_tip_candidates(x_A, y_A, slope, arrow_size):
    """Coordinates of the arrow tip offset from (x_A, y_A)."""
    return [
        [
            ((slope**2 + 1) * x_A - np.sqrt(slope**2 + 1) * arrow_size) / (slope**2 + 1),
            -((np.sqrt(slope**2 + 1) * arrow_size * slope - (slope**2 + 1) * y_A) / (slope**2 + 1)),
        ],
        [
            ((slope**2 + 1) * x_A + np.sqrt(slope**2 + 1) * arrow_size) / (slope**2 + 1),
            ((np.sqrt(slope**2 + 1) * arrow_size * slope + (slope**2 + 1) * y_A) / (slope**2 + 1)),
        ],
    ]


def _arrow_base_candidates(x_F, y_F, slope, arrow_width):
    """Coordinates of the arrow base points around (x_F, y_F)."""
    return [
        [
            ((slope**2 + 1) * x_F - np.sqrt(slope**2 + 1) * arrow_width) / (slope**2 + 1),
            -((np.sqrt(slope**2 + 1) * arrow_width * slope - (slope**2 + 1) * y_F) / (slope**2 + 1)),
        ],
        [
            ((slope**2 + 1) * x_F + np.sqrt(slope**2 + 1) * arrow_width) / (slope**2 + 1),
            ((np.sqrt(slope**2 + 1) * arrow_width * slope + (slope**2 + 1) * y_F) / (slope**2 + 1)),
        ],
    ]


def _arrow_side_candidates(x_F, y_F, intercept, slope, arrow_width):
    """Coordinates of the lateral arrow-head points."""
    root = np.sqrt(
        -slope**2 * y_F**2
        + (arrow_width**2 - intercept**2) * slope**2
        + 2 * intercept * slope * x_F
        + arrow_width**2
        - x_F**2
        + 2 * (intercept * slope**2 - slope * x_F) * y_F
    )
    return [
        [
            (slope**2 * x_F + intercept * slope - slope * y_F - root * slope) / (slope**2 + 1),
            (intercept * slope**2 - slope * x_F + y_F + root) / (slope**2 + 1),
        ],
        [
            (slope**2 * x_F + intercept * slope - slope * y_F + root * slope) / (slope**2 + 1),
            (intercept * slope**2 - slope * x_F + y_F - root) / (slope**2 + 1),
        ],
    ]


def _draw_arrow_2d(start_point, end_point, fig, ax, arrow_size, arrow_width, arrow_color, zorder):
    """Draw the original custom 2D polygon arrow."""
    x0 = start_point
    xA = end_point
    xB = [0, 0]
    xF = [0, 0]
    if x0[0] == xA[0]:
        xB[0] = xA[0]
        xF[0] = xA[0]
        if x0[1] >= xA[1]:
            xF[1] = arrow_size + xA[1]
            xB[1] = -arrow_width + xF[1]
        else:
            xF[1] = -arrow_size + xA[1]
            xB[1] = arrow_width + xF[1]
        xC = [xF[0] - arrow_width, xF[1]]
        xD = [xF[0] + arrow_width, xF[1]]
    elif x0[1] == xA[1]:
        xF[1] = xA[1]
        xB[1] = xA[1]
        if x0[0] >= xA[0]:
            xF[0] = arrow_size + xA[0]
            xB[0] = -arrow_width + xF[0]
        else:
            xF[0] = -arrow_size + xA[0]
            xB[0] = arrow_width + xF[0]
        xC = [xF[0], xF[1] - arrow_width]
        xD = [xF[0], xF[1] + arrow_width]
    elif xA[0] > x0[0]:
        slope = (xA[1] - x0[1]) / (xA[0] - x0[0])
        xF = _arrow_tip_candidates(xA[0], xA[1], slope, arrow_size)[0]
        xB = _arrow_base_candidates(xF[0], xF[1], slope, arrow_width)[1]
        intercept = (1 / slope) * xF[0] + xF[1]
        xC, xD = _arrow_side_candidates(xF[0], xF[1], intercept, slope, arrow_width)
    elif xA[0] < x0[0]:
        slope = (xA[1] - x0[1]) / (xA[0] - x0[0])
        xF = _arrow_tip_candidates(xA[0], xA[1], slope, arrow_size)[1]
        xB = _arrow_base_candidates(xF[0], xF[1], slope, arrow_width)[0]
        intercept = (1 / slope) * xF[0] + xF[1]
        xC, xD = _arrow_side_candidates(xF[0], xF[1], intercept, slope, arrow_width)
    else:
        return []

    shaft = ax.plot(
        [x0[0], xA[0]],
        [x0[1], xA[1]],
        color=arrow_color,
        zorder=zorder,
        clip_on=False,
    )
    head = Polygon([xA, xC, xB, xD])
    patch = PatchCollection(
        [head],
        facecolor=arrow_color,
        edgecolor=arrow_color,
        alpha=1,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_collection(patch)
    return shaft + [head]


def _draw_arrow_3d(start_point, end_point, fig, ax, arrow_size, arrow_width, arrow_color, zorder):
    """Draw a 3D cone aligned with the local trajectory direction."""
    if arrow_size <= 0 or arrow_width <= 0:
        return []

    start = np.asarray(start_point, dtype=float)
    tip = np.asarray(end_point, dtype=float)
    direction = tip - start
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return []

    unit_direction = direction / norm
    cone_base = tip - arrow_size * unit_direction

    shaft = ax.plot(
        [start[0], cone_base[0]],
        [start[1], cone_base[1]],
        [start[2], cone_base[2]],
        color=arrow_color,
        zorder=zorder,
    )

    basis_1, basis_2 = _orthonormal_basis_perpendicular_to(unit_direction)
    theta = np.linspace(0, 2 * np.pi, 18)
    height = np.linspace(0, arrow_size, 8)
    theta_grid, height_grid = np.meshgrid(theta, height)
    radius_grid = arrow_width * (height_grid / arrow_size)

    cone = (
        tip
        - height_grid[..., None] * unit_direction
        + radius_grid[..., None]
        * (
            np.cos(theta_grid)[..., None] * basis_1
            + np.sin(theta_grid)[..., None] * basis_2
        )
    )
    surface = ax.plot_surface(
        cone[..., 0],
        cone[..., 1],
        cone[..., 2],
        color=arrow_color,
        alpha=1,
        linewidth=0,
        shade=True,
        zorder=zorder,
    )
    return shaft + [surface]


def _orthonormal_basis_perpendicular_to(vector):
    """Return two perpendicular unit vectors spanning the plane normal to vector."""
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(vector, reference)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])

    basis_1 = np.cross(vector, reference)
    basis_1 = basis_1 / np.linalg.norm(basis_1)
    basis_2 = np.cross(vector, basis_1)
    basis_2 = basis_2 / np.linalg.norm(basis_2)
    return basis_1, basis_2


def draw_state_space(strategy_labels, payoff_data, ax, font_size, zorder):
    """Draw the state-space frame for a supported game.

    Parameters
    ----------
    strategy_labels : sequence of str
        Labels shown on simplex vertices or cube axes.
    payoff_data : numpy.ndarray or tuple of numpy.ndarray
        Payoff representation used to infer the game class.
    ax : matplotlib axes
        Axes on which the state space is drawn. Must be a 3D axes for 1Pop4S and
        3Pop2S games.
    font_size : float
        Font size for labels.
    zorder : float
        Drawing order of the state-space frame.

    Returns
    -------
    list
        Matplotlib artists created by the function.
    """
    game_class = infer_game_class(payoff_data)
    if game_class == "1Pop3S":
        strategy_1_vertex = simplex_to_plane_2p3s(1, 0)
        strategy_2_vertex = simplex_to_plane_2p3s(0, 1)
        strategy_3_vertex = simplex_to_plane_2p3s(0, 0)
        strategy_1_label = ax.annotate(strategy_labels[0], (strategy_1_vertex[0], strategy_1_vertex[1] + 0.04), ha="center", fontsize=font_size, zorder=zorder)
        strategy_2_label = ax.annotate(strategy_labels[1], (strategy_2_vertex[0], strategy_2_vertex[1] - 0.08), ha="center", fontsize=font_size, zorder=zorder)
        strategy_3_label = ax.annotate(strategy_labels[2], (strategy_3_vertex[0], strategy_3_vertex[1] - 0.08), ha="center", fontsize=font_size, zorder=zorder)
        edge_endpoints = (
            (strategy_1_vertex, strategy_2_vertex),
            (strategy_1_vertex, strategy_3_vertex),
            (strategy_2_vertex, strategy_3_vertex),
        )
        lines = []
        for start, end in edge_endpoints:
            lines += ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color='black',
                zorder=zorder,
                alpha=1,
                clip_on=False,
            )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.10, (3 ** 0.5) / 2 + 0.05)
        ax.set_aspect('equal', adjustable='box')
        return lines + [strategy_1_label, strategy_2_label, strategy_3_label]
    if game_class == "2Pop2S":
        ax.set_xlabel(strategy_labels[0], fontsize=font_size)
        ax.set_ylabel(strategy_labels[1], fontsize=font_size)
        edges = [([0, 1], [0, 0]), ([1, 1], [0, 1]), ([1, 0], [1, 1]), ([0, 0], [1, 0])]
        lines = []
        for xs, ys in edges:
            lines += ax.plot(xs, ys, color='black', zorder=zorder, alpha=1, clip_on=False)
        return lines
    if game_class == "3Pop2S":
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel(strategy_labels[0], fontsize=font_size)
        ax.set_ylabel(strategy_labels[1], fontsize=font_size)
        ax.set_zlabel(strategy_labels[2], fontsize=font_size)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_zticks([0, 1])
        ax.set_xticklabels(["0", "1"], fontsize=font_size - 2)
        ax.set_yticklabels(["0", "1"], fontsize=font_size - 2)
        ax.set_zticklabels(["0", "1"], fontsize=font_size - 2)
        # Hide panes/axis lines but keep ticks/labels for a minimal look.
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor((1, 1, 1, 0))
        # Hide axis lines using _axinfo (mplot3d-friendly)
        for axis in ("xaxis", "yaxis", "zaxis"):
            try:
                getattr(ax, axis)._axinfo["axisline"]["linewidth"] = 0.0
            except Exception:
                pass
        corners = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ]
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        artists = []
        for start_idx, end_idx in edges:
            s = corners[start_idx]
            e = corners[end_idx]
            artists += ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color='black', zorder=zorder, alpha=1)
        return artists
    if game_class == "1Pop4S":
        strategy_1_vertex = simplex_to_plane_2p4s(1, 0, 0)
        strategy_2_vertex = simplex_to_plane_2p4s(0, 1, 0)
        strategy_3_vertex = simplex_to_plane_2p4s(0, 0, 1)
        strategy_4_vertex = simplex_to_plane_2p4s(0, 0, 0)
        ax.grid(False)
        strategy_1_label = ax.text(strategy_1_vertex[0], strategy_1_vertex[1] + 0.05, strategy_1_vertex[2], strategy_labels[0], fontsize=font_size, zorder=zorder)
        strategy_2_label = ax.text(strategy_2_vertex[0] - 0.05, strategy_2_vertex[1], strategy_2_vertex[2], strategy_labels[1], fontsize=font_size, zorder=zorder)
        strategy_3_label = ax.text(strategy_3_vertex[0] + 0.05, strategy_3_vertex[1] - 0.022, strategy_3_vertex[2], strategy_labels[2], fontsize=font_size, zorder=zorder)
        strategy_4_label = ax.text(strategy_4_vertex[0] - 0.02, strategy_4_vertex[1] - 0.022, strategy_4_vertex[2] + 0.05, strategy_labels[3], fontsize=font_size, zorder=zorder)
        edge_endpoints = (
            (strategy_1_vertex, strategy_2_vertex),
            (strategy_2_vertex, strategy_3_vertex),
            (strategy_3_vertex, strategy_1_vertex),
            (strategy_4_vertex, strategy_1_vertex),
            (strategy_4_vertex, strategy_2_vertex),
            (strategy_4_vertex, strategy_3_vertex),
        )
        lines = []
        for start, end in edge_endpoints:
            lines += ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color='black',
                zorder=zorder,
                alpha=1,
                clip_on=False,
            )
        return lines + [strategy_1_label, strategy_2_label, strategy_3_label, strategy_4_label]
    return []


def draw_3d_faces(payoff_data, ax, face_colors=None, alpha=0.15, zorder=0):
    """Draw transparent colored faces for 3D state spaces."""
    game_class = infer_game_class(payoff_data)
    if game_class == "1Pop4S":
        faces = _tetrahedron_faces_2p4s()
    elif game_class == "3Pop2S":
        faces = _cube_faces_3p2s()
    else:
        return []

    colors = _face_colors(face_colors, len(faces))
    artists = []
    for vertices, color in zip(faces, colors):
        collection = Poly3DCollection(
            [vertices],
            facecolors=mcolors.to_rgba(color, alpha),
            edgecolors="none",
            zorder=zorder,
        )
        ax.add_collection3d(collection)
        artists.append(collection)
    return artists


def _tetrahedron_faces_2p4s():
    strategy_1_vertex = simplex_to_plane_2p4s(1, 0, 0)
    strategy_2_vertex = simplex_to_plane_2p4s(0, 1, 0)
    strategy_3_vertex = simplex_to_plane_2p4s(0, 0, 1)
    strategy_4_vertex = simplex_to_plane_2p4s(0, 0, 0)
    return [
        [strategy_1_vertex, strategy_2_vertex, strategy_3_vertex],
        [strategy_1_vertex, strategy_2_vertex, strategy_4_vertex],
        [strategy_1_vertex, strategy_3_vertex, strategy_4_vertex],
        [strategy_2_vertex, strategy_3_vertex, strategy_4_vertex],
    ]


def _cube_faces_3p2s():
    return [
        [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    ]


def _face_colors(face_colors, face_count):
    default_colors = [
        "#9ecae1",
        "#fdd49e",
        "#a1d99b",
        "#fbb4b9",
        "#c7c7e2",
        "#fdae6b",
    ]
    if face_colors is None:
        return default_colors[:face_count]
    if isinstance(face_colors, str):
        return [face_colors] * face_count
    if len(face_colors) != face_count:
        raise ValueError(
            f"Expected {face_count} face colors, got {len(face_colors)}."
        )
    return list(face_colors)


def plot_trajectory(
    initial_state,
    payoff_data,
    time_step,
    arrow_positions,
    tmax,
    fig,
    ax,
    trajectory_color,
    trajectory_linewidth,
    arrow_size,
    arrow_width,
    zorder,
    arrow_color=None,
):
    """Draw forward and backward trajectories from one initial state.

    Parameters
    ----------
    initial_state : sequence of float
        Initial condition. For 2Pop2S and 3Pop2S games, entries are probabilities
        of the first listed strategy for each player. For 1Pop3S and 1Pop4S games,
        entries are reduced simplex coordinates.
    payoff_data : numpy.ndarray or tuple of numpy.ndarray
        Payoff representation used to infer the game class and compute the
        replicator dynamics.
    time_step : float
        Time step used for numerical integration.
    arrow_positions : sequence of float
        Fractions of the sampled forward trajectory where direction markers are
        drawn. Use an empty list to omit arrows.
    tmax : float
        Time horizon for both forward and backward integration.
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib axes
        Axes on which trajectories are drawn.
    trajectory_color : matplotlib color
        Color of the trajectory lines.
    trajectory_linewidth : float
        Width of the trajectory lines. The same width is used for forward and
        backward trajectories.
    arrow_size : float
        Size of trajectory direction markers.
    arrow_width : float
        Width of 2D arrow heads or radius of 3D cone markers.
    zorder : float
        Drawing order of trajectories and direction markers.
    arrow_color : matplotlib color, optional
        Color of direction markers. If None, `trajectory_color` is used.

    Returns
    -------
    list or None
        Matplotlib artists created by the function, or None if the game class is
        unsupported.
    """
    t = np.linspace(0, tmax, int(tmax / time_step))
    line_color = trajectory_color if trajectory_color is not None else 'black'
    arrow_col = arrow_color if arrow_color is not None else line_color
    game_class = infer_game_class(payoff_data)

    if game_class == "1Pop3S":
        x0, y0 = initial_state
        forward_solution = odeint(dynamics.replicator_2p3s, [x0, y0], t, (payoff_data,))
        backward_solution = odeint(dynamics.reverse_replicator_2p3s, [x0, y0], t, (payoff_data,))
        forward_path = np.asarray(
            [simplex_to_plane_2p3s(point[0], point[1]) for point in forward_solution]
        )
        backward_path = np.asarray(
            [simplex_to_plane_2p3s(point[0], point[1]) for point in backward_solution]
        )
        forward_line = ax.plot(
            forward_path[:, 0],
            forward_path[:, 1],
            linewidth=trajectory_linewidth,
            color=line_color,
            zorder=zorder,
            clip_on=False,
        )
        backward_line = ax.plot(
            backward_path[:, 0],
            backward_path[:, 1],
            linewidth=trajectory_linewidth,
            color=line_color,
            zorder=zorder,
            clip_on=False,
        )
        arrow_artists = []
        for frac in arrow_positions:
            base = min(max(int(frac * (len(forward_path) - 1)), 0), len(forward_path) - 2)
            arrow_artists += _draw_arrow_2d(
                forward_path[base],
                forward_path[base + 1],
                fig,
                ax,
                arrow_width=arrow_width,
                arrow_size=arrow_size,
                arrow_color=arrow_col,
                zorder=zorder,
            )
        return forward_line + backward_line + arrow_artists

    if game_class == "2Pop2S":
        x0, y0 = initial_state
        forward_solution = odeint(dynamics.replicator_2p2s, [x0, y0], t, (payoff_data,))
        backward_solution = odeint(dynamics.reverse_replicator_2p2s, [x0, y0], t, (payoff_data,))
        forward_line = ax.plot(
            forward_solution[:, 0],
            forward_solution[:, 1],
            linewidth=trajectory_linewidth,
            color=line_color,
            zorder=zorder,
            clip_on=False,
        )
        backward_line = ax.plot(
            backward_solution[:, 0],
            backward_solution[:, 1],
            linewidth=trajectory_linewidth,
            color=line_color,
            zorder=zorder,
            clip_on=False,
        )
        arrow_artists = []
        for frac in arrow_positions:
            base = min(max(int(frac * (len(forward_solution) - 1)), 0), len(forward_solution) - 2)
            arrow_artists += _draw_arrow_2d(
                forward_solution[base],
                forward_solution[base + 1],
                fig,
                ax,
                arrow_width=arrow_width,
                arrow_size=arrow_size,
                arrow_color=arrow_col,
                zorder=zorder,
            )
        return forward_line + backward_line + arrow_artists

    if game_class == "3Pop2S":
        x0, y0, z0 = initial_state
        forward_solution = odeint(dynamics.replicator_3p2s, [x0, y0, z0], t, (payoff_data,))
        backward_solution = odeint(dynamics.reverse_replicator_3p2s, [x0, y0, z0], t, (payoff_data,))
        forward_line = ax.plot(
            forward_solution[:, 0],
            forward_solution[:, 1],
            forward_solution[:, 2],
            linewidth=trajectory_linewidth,
            color=line_color,
            zorder=zorder,
        )
        backward_line = ax.plot(
            backward_solution[:, 0],
            backward_solution[:, 1],
            backward_solution[:, 2],
            linewidth=trajectory_linewidth,
            color=line_color,
            zorder=zorder,
        )
        arrow_artists = []
        for frac in arrow_positions:
            base = min(max(int(frac * (len(forward_solution) - 1)), 0), len(forward_solution) - 2)
            arrow_artists += _draw_arrow_3d(
                forward_solution[base],
                forward_solution[base + 1],
                fig,
                ax,
                arrow_width=arrow_width,
                arrow_size=arrow_size,
                arrow_color=arrow_col,
                zorder=zorder,
            )
        return forward_line + backward_line + arrow_artists

    if game_class == "1Pop4S":
        x0, y0, z0 = initial_state
        forward_solution = odeint(dynamics.replicator_2p4s, [x0, y0, z0], t, (payoff_data,))
        backward_solution = odeint(dynamics.reverse_replicator_2p4s, [x0, y0, z0], t, (payoff_data,))
        forward_path = np.asarray(
            [simplex_to_plane_2p4s(point[0], point[1], point[2]) for point in forward_solution]
        )
        backward_path = np.asarray(
            [simplex_to_plane_2p4s(point[0], point[1], point[2]) for point in backward_solution]
        )
        forward_line = ax.plot(
            forward_path[:, 0],
            forward_path[:, 1],
            forward_path[:, 2],
            linewidth=trajectory_linewidth,
            color=line_color,
            zorder=zorder,
        )
        backward_line = ax.plot(
            backward_path[:, 0],
            backward_path[:, 1],
            backward_path[:, 2],
            linewidth=trajectory_linewidth,
            color=line_color,
            zorder=zorder,
        )
        arrow_artists = []
        for frac in arrow_positions:
            base = min(max(int(frac * (len(forward_path) - 1)), 0), len(forward_path) - 2)
            arrow_artists += _draw_arrow_3d(
                forward_path[base],
                forward_path[base + 1],
                fig,
                ax,
                arrow_width=arrow_width,
                arrow_size=arrow_size,
                arrow_color=arrow_col,
                zorder=zorder,
            )
        return forward_line + backward_line + arrow_artists

    return None


def plot_equilibria(
    payoff_data,
    ax,
    sink_color,
    saddle_color,
    source_color,
    equilibrium_size,
    zorder,
    center_color=None,
    equilibrium_edgecolor="black",
):
    """Classify and plot isolated equilibria.

    Parameters
    ----------
    payoff_data : numpy.ndarray or tuple of numpy.ndarray
        Payoff representation of a supported game.
    ax : matplotlib axes
        Axes on which equilibrium markers are drawn.
    sink_color, saddle_color, source_color : matplotlib color
        Marker colors for sink, saddle, and source equilibria. Equilibria
        classified as "unstable" are drawn with `source_color`.
    equilibrium_size : float
        Marker size in 2D. In 3D, this controls sphere radius indirectly.
    zorder : float
        Drawing order of equilibrium markers.
    center_color : matplotlib color or None, optional
        Marker color for centers. If None, centers use `source_color`.
    equilibrium_edgecolor : matplotlib color or None, default="black"
        Outline color for 2D equilibrium markers. If None, markers are drawn
        without outlines. 3D equilibria are drawn as shaded spheres and do not
        use a separate marker outline.

    Returns
    -------
    list
        Five lists of equilibrium positions grouped as source, saddle, sink,
        center, and undetermined.

    Warns
    -----
    PlottingWarning
        Raised when "unstable" equilibria are drawn with the source color.
    """
    source, sink, saddle, center, undetermined = [], [], [], [], []
    center_color = source_color if center_color is None else center_color

    result = analysis.analyze_equilibria(payoff_data)
    unstable_plotted_as_source = False

    def _point_to_plot(equilibrium):
        if result.game_class == "1Pop3S":
            return np.array(simplex_to_plane_2p3s(equilibrium.reduced_position[0], equilibrium.reduced_position[1]))
        if result.game_class == "1Pop4S":
            return np.array(
                simplex_to_plane_2p4s(
                    equilibrium.reduced_position[0],
                    equilibrium.reduced_position[1],
                    equilibrium.reduced_position[2],
                )
            )
        return np.asarray(equilibrium.full_position, dtype=float)

    for equilibrium in result.equilibria:
        point_to_plot = _point_to_plot(equilibrium)
        if equilibrium.stability == 'sink':
            sink.append(point_to_plot)
        elif equilibrium.stability == 'source':
            source.append(point_to_plot)
        elif equilibrium.stability == 'unstable':
            source.append(point_to_plot)
            unstable_plotted_as_source = True
        elif equilibrium.stability == 'saddle':
            saddle.append(point_to_plot)
        elif equilibrium.stability == 'center':
            center.append(point_to_plot)
        else:
            undetermined.append(point_to_plot)

    if unstable_plotted_as_source:
        warnings.warn(
            "At least one equilibrium is classified as 'unstable': linearization "
            "proves it is not stable, but does not distinguish source from saddle. "
            "For plotting compatibility, unstable equilibria are drawn with the "
            "source color and returned in the source bucket.",
            PlottingWarning,
            stacklevel=2,
        )

    def _to_raw(point):
        if result.game_class == "1Pop3S":
            r, p = plane_to_simplex_2p3s(point[0], point[1])
            return [r, p, 1 - r - p]
        if result.game_class == "2Pop2S":
            return point.tolist()
        if result.game_class in ("3Pop2S", "1Pop4S"):
            return point.tolist()
        return point.tolist()

    def _sphere_radius(size):
        return np.clip(np.sqrt(size) / 200.0, 0.01, 0.08) * (2.0 / 3.0)

    def _plot_spheres(points, color, alpha=0.85):
        if points.size == 0:
            return
        radius = _sphere_radius(equilibrium_size)
        rgba = mcolors.to_rgba(color, alpha=alpha)
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 12)
        for center in points:
            cx, cy, cz = center
            X = radius * np.outer(np.cos(u), np.sin(v)) + cx
            Y = radius * np.outer(np.sin(u), np.sin(v)) + cy
            Z = radius * np.outer(np.ones_like(u), np.cos(v)) + cz
            facecolors = np.empty(X.shape + (4,))
            facecolors[..., :] = rgba
            ax.plot_surface(
                X,
                Y,
                Z,
                rstride=1,
                cstride=1,
                facecolors=facecolors,
                linewidth=0,
                antialiased=True,
                shade=True,
                zorder=zorder,
            )

    def _scatter(points, color, marker='o'):
        if not points:
            return
        pts = np.array(points)
        if result.game_class in ("3Pop2S", "1Pop4S"):
            _plot_spheres(pts, color)
        else:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=equilibrium_size,
                color=color,
                marker=marker,
                edgecolors=equilibrium_edgecolor if equilibrium_edgecolor is not None else "none",
                alpha=1,
                zorder=zorder,
                clip_on=False,
            )

    def _scatter_undetermined(points):
        if not points:
            return
        pts = np.array(points)
        if result.game_class in ("3Pop2S", "1Pop4S"):
            _plot_spheres(pts, 'gray', alpha=0.45)
        else:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=equilibrium_size,
                facecolors='none',
                edgecolors=equilibrium_edgecolor if equilibrium_edgecolor is not None else "none",
                alpha=1,
                zorder=zorder,
                clip_on=False,
            )

    raw_source = [_to_raw(pt) for pt in source]
    raw_saddle = [_to_raw(pt) for pt in saddle]
    raw_sink = [_to_raw(pt) for pt in sink]
    raw_center = [_to_raw(pt) for pt in center]
    raw_undetermined = [_to_raw(pt) for pt in undetermined]

    _scatter(source, source_color)
    _scatter(saddle, saddle_color)
    _scatter(sink, sink_color)
    _scatter(center, center_color)
    _scatter_undetermined(undetermined)

    return [raw_source, raw_saddle, raw_sink, raw_center, raw_undetermined]


def plot_vector_field(
    payoff_data,
    ax,
    grid=15,
    margin=0.02,
    color="black",
    alpha=0.75,
    length=0.04,
    width=0.003,
    zorder=15,
    normalize=True,
):
    """Plot a sparse vector field for a supported game.

    Parameters
    ----------
    payoff_data : numpy.ndarray or tuple of numpy.ndarray
        Payoff representation of a supported game.
    ax : matplotlib axes
        Axes on which the vector field is drawn.
    grid : int, default=15
        Grid density used to sample the state space.
    margin : float, default=0.02
        Distance from the boundary used when sampling vector-field points.
    color : matplotlib color, default="black"
        Color of vector-field arrows.
    alpha : float, default=0.75
        Transparency of vector-field arrows.
    length : float, default=0.04
        Arrow length when vectors are normalized.
    width : float, default=0.003
        Width of 2D arrows. Matplotlib's 3D quiver does not use this parameter
        in the same way.
    zorder : float, default=15
        Drawing order of vector-field arrows.
    normalize : bool, default=True
        If True, arrows show direction only. If False, arrow lengths reflect
        vector-field magnitude.

    Returns
    -------
    matplotlib artist or None
        Quiver artist, or None if no nonzero vectors are available.
    """
    game_class = infer_game_class(payoff_data)

    if game_class == "2Pop2S":
        points, vectors = _vector_field_2p2s(payoff_data, grid, margin)
        return _quiver_2d(ax, points, vectors, color, alpha, length, width, zorder, normalize)

    if game_class == "1Pop3S":
        points, vectors = _vector_field_2p3s(payoff_data, grid, margin)
        return _quiver_2d(ax, points, vectors, color, alpha, length, width, zorder, normalize)

    if game_class == "3Pop2S":
        points, vectors = _vector_field_3p2s(payoff_data, grid, margin)
        return _quiver_3d(ax, points, vectors, color, alpha, length, zorder, normalize)

    if game_class == "1Pop4S":
        points, vectors = _vector_field_2p4s(payoff_data, grid, margin)
        return _quiver_3d(ax, points, vectors, color, alpha, length, zorder, normalize)

    raise ValueError(f"Unsupported game class: {game_class}")


def _plot_edge_flow(
    *, game_class, payoff_data, fig, ax, show_arrows,
    trajectory_color, trajectory_linewidth, arrow_size, arrow_width, zorder,
):
    """Draw exact one-dimensional flow on invariant edges of 2D state spaces."""
    if game_class == "1Pop3S":
        matrix = np.asarray(payoff_data, dtype=float)
        edge_specs = []
        for first, second in ((0, 1), (0, 2), (1, 2)):
            edge_matrix = matrix[np.ix_([first, second], [first, second])]

            def coordinates(frequency, first=first, second=second):
                state = np.zeros(3)
                state[first] = frequency
                state[second] = 1 - frequency
                return np.asarray(simplex_to_plane_2p3s(state[0], state[1]))

            edge_specs.append((edge_matrix, coordinates))
    else:
        player_1, player_2 = (np.asarray(matrix, dtype=float) for matrix in payoff_data)
        edge_specs = []
        for opponent_frequency in (0.0, 1.0):
            fitness = player_1 @ np.array([opponent_frequency, 1 - opponent_frequency])
            edge_matrix = np.repeat(fitness[:, None], 2, axis=1)
            edge_specs.append(
                (edge_matrix, lambda frequency, fixed=opponent_frequency: np.array([frequency, fixed]))
            )
        for opponent_frequency in (0.0, 1.0):
            fitness = player_2 @ np.array([opponent_frequency, 1 - opponent_frequency])
            edge_matrix = np.repeat(fitness[:, None], 2, axis=1)
            edge_specs.append(
                (edge_matrix, lambda frequency, fixed=opponent_frequency: np.array([fixed, frequency]))
            )

    for edge_matrix, coordinates in edge_specs:
        points, continuum = dynamics._analyze_1d(edge_matrix)
        start, end = coordinates(0), coordinates(1)
        ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            color=trajectory_color, linewidth=trajectory_linewidth,
            zorder=zorder, clip_on=False,
        )
        if continuum or not show_arrows:
            continue

        positions = [
            (left.x + right.x) / 2
            for left, right in zip(points[:-1], points[1:])
        ]
        for frequency in positions:
            direction = np.sign(dynamics.replicator_1d(frequency, edge_matrix))
            if direction == 0:
                continue
            _draw_arrow_2d(
                coordinates(frequency - direction * 0.01),
                coordinates(frequency),
                fig,
                ax,
                arrow_width=arrow_width,
                arrow_size=arrow_size,
                arrow_color=trajectory_color,
                zorder=zorder,
            )


def phase_portrait(
    game,
    *,
    fig=None,
    ax=None,
    figsize=None,
    view_elev=DEFAULT_PLOT_STYLE["view_elev"],
    view_azim=DEFAULT_PLOT_STYLE["view_azim"],
    xlabel=None,
    ylabel=None,
    zlabel=None,
    title_pad=18,
    starts=None,
    random_state=None,
    trajectory_number=DEFAULT_PLOT_STYLE["trajectory_number"],
    show_edge_flow=DEFAULT_PLOT_STYLE["show_edge_flow"],
    simplex_font_size=DEFAULT_PLOT_STYLE["simplex_font_size"],
    simplex_zorder=DEFAULT_PLOT_STYLE["simplex_zorder"],
    show_speed=DEFAULT_PLOT_STYLE["show_speed"],
    speed_grid=DEFAULT_PLOT_STYLE["speed_grid"],
    speed_cmap=DEFAULT_PLOT_STYLE["speed_cmap"],
    speed_levels=DEFAULT_PLOT_STYLE["speed_levels"],
    speed_zorder=DEFAULT_PLOT_STYLE["speed_zorder"],
    show_vector_field=DEFAULT_PLOT_STYLE["show_vector_field"],
    vector_grid=DEFAULT_PLOT_STYLE["vector_grid"],
    vector_margin=DEFAULT_PLOT_STYLE["vector_margin"],
    vector_color=DEFAULT_PLOT_STYLE["vector_color"],
    vector_alpha=DEFAULT_PLOT_STYLE["vector_alpha"],
    vector_length=DEFAULT_PLOT_STYLE["vector_length"],
    vector_width=DEFAULT_PLOT_STYLE["vector_width"],
    vector_zorder=DEFAULT_PLOT_STYLE["vector_zorder"],
    vector_normalize=DEFAULT_PLOT_STYLE["vector_normalize"],
    show_faces=DEFAULT_PLOT_STYLE["show_faces"],
    face_colors=DEFAULT_PLOT_STYLE["face_colors"],
    face_alpha=DEFAULT_PLOT_STYLE["face_alpha"],
    face_zorder=DEFAULT_PLOT_STYLE["face_zorder"],
    show_trajectories=DEFAULT_PLOT_STYLE["show_trajectories"],
    trajectory_step=DEFAULT_PLOT_STYLE["trajectory_step"],
    trajectory_arrows=None,
    tmax=DEFAULT_PLOT_STYLE["tmax"],
    trajectory_color=DEFAULT_PLOT_STYLE["trajectory_color"],
    trajectory_linewidth=None,
    arrow_size=DEFAULT_PLOT_STYLE["arrow_size"],
    arrow_width=DEFAULT_PLOT_STYLE["arrow_width"],
    trajectory_zorder=DEFAULT_PLOT_STYLE["trajectory_zorder"],
    show_equilibria=DEFAULT_PLOT_STYLE["show_equilibria"],
    sink_color=DEFAULT_PLOT_STYLE["sink_color"],
    saddle_color=DEFAULT_PLOT_STYLE["saddle_color"],
    source_color=DEFAULT_PLOT_STYLE["source_color"],
    center_color=DEFAULT_PLOT_STYLE["center_color"],
    equilibrium_size=DEFAULT_PLOT_STYLE["equilibrium_size"],
    equilibrium_edgecolor=DEFAULT_PLOT_STYLE["equilibrium_edgecolor"],
    equilibrium_zorder=DEFAULT_PLOT_STYLE["equilibrium_zorder"],
    continuum_color="tab:purple",
):
    """Plot replicator dynamics for a supported game.

    This is the main plotting interface of pyNamo. It draws the appropriate
    state space for the game class and can optionally add a speed field, vector
    field, trajectories, and equilibria.

    A single symmetric 2x2 matrix selects a one-population phase line
    (1Pop2S), with x the first strategy's frequency. Endpoint labels come from
    the game. xlabel is optional; ylabel and zorder controls have no effect.
    Layering is fixed: frame, arrowheads, then equilibrium markers.
    trajectory_arrows=None places centered heads between equilibria; a list
    supplies frequencies in (0,1), and [] hides heads. Arrow positions cannot
    be equilibria. Arrow size/width are display-scaled in 1D (defaults 10-point
    length and 3.75-point half-width). The line defaults to 1.2-point thickness.
    show_equilibria=False hides markers but analysis still positions arrows.
    continuum_color colors an entire stationary interval without arrows.
    Integration, speed/vector field, and 3D options do not apply to phase lines.

    Parameters
    ----------
    game : pn.Game
        Game object to plot. The game class is inferred from the payoff data.
        Supported classes are "1Pop2S", "2Pop2S", "1Pop3S", "1Pop4S", and "3Pop2S".
    fig : matplotlib.figure.Figure, optional
        Existing Matplotlib figure. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axes. If None, new axes are created. A 3D axis is
        created automatically for 1Pop4S and 3Pop2S games.
    figsize : tuple, default=(6, 6)
        Size of the figure when a new figure is created.
    view_elev : float, default=25
        Elevation angle for 3D plots. Ignored for 2D plots.
    view_azim : float, default=35
        Azimuth angle for 3D plots. Ignored for 2D plots.
    xlabel, ylabel, zlabel : str or None, optional
        Axis-label overrides. If None, pyNamo uses labels derived from the game.
        `zlabel` is ignored for 2D plots.
    title_pad : float, default=18
        Padding between the plot title and the axes. Increasing this is useful
        when title text is close to simplex labels.
    starts : list of list of float, optional
        Initial conditions for trajectories. If None, deterministic interior
        automatically spaced interior points are generated. All supplied
        coordinates must lie between 0 and 1. Supplying starts overrides this
        generation.

        For 2Pop2S games, each start is [x, y], where x is the probability that
        player 1 uses their first listed strategy and y is the probability that
        player 2 uses their first listed strategy.

        For 3Pop2S games, each start is [x, y, z], where each coordinate is the
        probability that the corresponding player uses their first listed
        strategy.

        For 1Pop3S games, each start is [x1, x2], the first two coordinates of
        the population state. The third coordinate is 1 - x1 - x2, so starts
        must also satisfy x1 + x2 <= 1.

        For 1Pop4S games, each start is [x1, x2, x3], the first three coordinates
        of the population state. The fourth coordinate is 1 - x1 - x2 - x3, so
        starts must also satisfy x1 + x2 + x3 <= 1.
    random_state : int or None, optional
        If supplied while `starts` is None, use the former random-start behavior
        with this seed instead of the deterministic grid.
    trajectory_number : int, default=4
        Exact number of deterministic initial conditions generated when
        `starts` is None. pyNamo recursively partitions a simplex or product of
        simplices into approximately equal regions and starts one trajectory at
        each region's centroid.
    show_edge_flow : bool or None, optional
        Draw invariant edges as one-dimensional phase lines, with arrows midway
        between consecutive edge equilibria. None enables this by default for
        1Pop3S and 2Pop2S only when starts are generated automatically. Explicit
        True enables it for those classes with user-supplied starts; False hides
        it. Edge flow is not drawn automatically for 3D classes.
    simplex_font_size : float, default=13
        Font size for simplex or cube labels.
    simplex_zorder : float, default=30
        Drawing order of the state-space frame and labels.
    show_speed : bool, default=True
        Whether to draw the speed field. Speed fields are currently drawn only
        for 2D state spaces: 2Pop2S and 1Pop3S.
    speed_grid : int, default=60
        Grid density used to compute the speed field.
    speed_cmap : matplotlib colormap, default=plt.cm.Spectral_r
        Colormap used for the speed field.
    speed_levels : int, default=12
        Number of contour levels in the speed field.
    speed_zorder : float, default=10
        Drawing order of the speed field.
    show_vector_field : bool, default=False
        Whether to draw a sparse vector field.
    vector_grid : int, default=15
        Grid density used to compute vector-field arrows.
    vector_margin : float, default=0.02
        Margin from the boundary when sampling vector-field points.
    vector_color : matplotlib color, default="black"
        Color of vector-field arrows.
    vector_alpha : float, default=0.75
        Transparency of vector-field arrows.
    vector_length : float, default=0.04
        Length scale for normalized vector-field arrows.
    vector_width : float, default=0.003
        Width of 2D vector-field arrows. Matplotlib's 3D quiver does not use
        this parameter in the same way.
    vector_zorder : float, default=15
        Drawing order of vector-field arrows.
    vector_normalize : bool, default=True
        If True, vector-field arrows show direction only and are rescaled to a
        common length. If False, arrow lengths reflect the magnitude of the
        vector field.
    show_faces : bool, default=False
        Whether to color the faces of 3D state spaces. Only affects 1Pop4S
        tetrahedra and 3Pop2S cubes.
    face_colors : matplotlib color, list of colors, or None, optional
        Face colors used when `show_faces=True`. If None, pyNamo uses a soft
        default palette. A single color applies to every face. A list must have
        four colors for 1Pop4S games or six colors for 3Pop2S games.
    face_alpha : float, default=0.15
        Transparency of colored 3D faces.
    face_zorder : float, default=0
        Drawing order of colored 3D faces.
    show_trajectories : bool, default=True
        Whether to draw trajectories.
    trajectory_step : float, default=0.02
        Time step used for numerical integration.
    trajectory_arrows : list of float or None, optional
        Positions at which to draw direction markers on forward trajectories.
        Values are fractions of the sampled trajectory, between 0 and 1. Thus the exact time at which the i_th arrow is drawn is trajectory_arrows[i]*tmax. Use an
        empty list `[]` to draw trajectories without arrows. If None, the
        default arrow positions are used. Boundary-flow arrows are always placed
        halfway between consecutive edge equilibria; their positions are not
        taken from this time-based parameter. An empty list `[]` hides arrows on
        both interior trajectories and boundary flows.
    tmax : float, default=45
        Time horizon for trajectory integration. pyNamo integrates both forward
        and backward trajectories from each initial condition.
    trajectory_color : matplotlib color or list of colors, default="black"
        Color of trajectories. A single color applies to all trajectories. A
        list assigns one color per trajectory and must have the same length as
        `starts`.
    trajectory_linewidth : float, default=1.2
        Width of trajectory lines. Increase
        this for denser or publication figures; decrease it when combining many
        trajectories with speed or vector fields.
    arrow_size : float, default=0.04
        Size of trajectory direction markers. In 2D this controls the length of
        the custom polygon arrow head. In 3D this controls the length of the
        cone marker.
    arrow_width : float, default=0.015
        Width of trajectory direction markers. In 2D this controls the width of
        the custom polygon arrow head. In 3D this controls the radius of the
        cone marker.
    trajectory_zorder : float, default=20
        Drawing order of trajectories and their direction markers.
    show_equilibria : bool, default=True
        Whether to compute and draw isolated equilibria.
    sink_color : matplotlib color, default="black"
        Color used for sink equilibria.
    saddle_color : matplotlib color, default="gray"
        Color used for saddle equilibria.
    source_color : matplotlib color, default="white"
        Color used for source equilibria. Equilibria classified as "unstable"
        are also drawn with this color.
    center_color : matplotlib color or None, optional
        Color used for centers. If None, centers use `source_color`.
    equilibrium_size : float, default=80
        Size of equilibrium markers. For 2D plots this is the scatter marker
        size. For 3D plots this determines the radius of the equilibrium
        spheres.
    equilibrium_edgecolor : matplotlib color or None, default="black"
        Outline color for 2D equilibrium markers. If None, markers are drawn
        without outlines. 3D equilibria are drawn as shaded spheres and do not
        use a separate marker outline.
    equilibrium_zorder : float, default=40
        Drawing order of equilibrium markers.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the plot. This is a 3D axes object for 1Pop4S and 3Pop2S
        games.

    Raises
    ------
    ValueError
        If the game does not belong to a supported game class.

    Notes
    -----
    Equilibrium stability is computed from the linearization restricted to
    admissible directions in the state space. Higher-dimensional non-isolated
    equilibrium manifolds are not plotted automatically. When stability cannot be classified
    conclusively, pyNamo emits a warning rather than forcing a classification.
    """
    payoff_data = _payoff_data(game)
    game_class = require_supported_game_class(game)
    if game_class == "1Pop2S":
        from ._one_dimensional import _phase_portrait_1d
        return _phase_portrait_1d(
            game, fig=fig, ax=ax, figsize=figsize or (7, 2.2), xlabel=xlabel,
            title_pad=title_pad, simplex_font_size=simplex_font_size,
            trajectory_color=trajectory_color,
            trajectory_linewidth=DEFAULT_PLOT_STYLE["trajectory_linewidth"] if trajectory_linewidth is None else trajectory_linewidth,
            trajectory_arrows=trajectory_arrows, show_trajectories=show_trajectories,
            arrow_size=arrow_size, arrow_width=arrow_width,
            show_equilibria=show_equilibria, equilibrium_size=equilibrium_size,
            sink_color=sink_color, source_color=source_color,
            equilibrium_edgecolor=equilibrium_edgecolor, continuum_color=continuum_color,
        )
    figsize = DEFAULT_PLOT_STYLE["figsize"] if figsize is None else figsize
    trajectory_linewidth = DEFAULT_PLOT_STYLE["trajectory_linewidth"] if trajectory_linewidth is None else trajectory_linewidth
    labels = _strategy_labels(game, game_class)
    player_strategy_labels = _player_strategy_labels(game, game_class)
    player_labels = _player_labels(game, game_class)
    title = getattr(game, "name", None)
    edge_arrow_positions = trajectory_arrows
    trajectory_arrows = (
        list(DEFAULT_PLOT_STYLE["trajectory_arrows"])
        if trajectory_arrows is None
        else trajectory_arrows
    )

    fig, ax = _get_or_create_axes(fig, ax, game_class, figsize, view_elev, view_azim)

    if show_faces and game_class in ("1Pop4S", "3Pop2S"):
        draw_3d_faces(
            payoff_data,
            ax,
            face_colors=face_colors,
            alpha=face_alpha,
            zorder=face_zorder,
        )

    draw_state_space(labels, payoff_data, ax, simplex_font_size, simplex_zorder)
    _set_default_axes_style(ax, game_class, labels, player_strategy_labels, player_labels)
    _apply_axis_label_overrides(ax, xlabel, ylabel, zlabel)

    if show_speed and game_class in ("2Pop2S", "1Pop3S"):
        x_region, y_region = _speed_regions(game_class)
        plot_speed_field(
            x_region=x_region,
            y_region=y_region,
            step=speed_grid,
            ax=ax,
            payoff_data=payoff_data,
            cmap=speed_cmap,
            levels=speed_levels,
            zorder=speed_zorder,
        )

    if show_vector_field:
        plot_vector_field(
            payoff_data,
            ax,
            grid=vector_grid,
            margin=vector_margin,
            color=vector_color,
            alpha=vector_alpha,
            length=vector_length,
            width=vector_width,
            zorder=vector_zorder,
            normalize=vector_normalize,
        )

    if show_equilibria:
        plot_equilibria(
            payoff_data,
            ax,
            sink_color=sink_color,
            saddle_color=saddle_color,
            source_color=source_color,
            center_color=center_color,
            equilibrium_size=equilibrium_size,
            equilibrium_edgecolor=equilibrium_edgecolor,
            zorder=equilibrium_zorder,
        )

    if show_trajectories:
        generated_starts = starts is None
        starts = (
            _default_starts(game_class, random_state, trajectory_number)
            if generated_starts
            else starts
        )
        trajectory_colors = _trajectory_colors(trajectory_color, len(starts))
        for start, color in zip(starts, trajectory_colors):
            plot_trajectory(
                start,
                payoff_data,
                time_step=trajectory_step,
                arrow_positions=trajectory_arrows,
                tmax=tmax,
                fig=fig,
                ax=ax,
                trajectory_color=color,
                trajectory_linewidth=trajectory_linewidth,
                arrow_size=arrow_size,
                arrow_width=arrow_width,
                zorder=trajectory_zorder,
                arrow_color=color,
            )

        edge_flow_enabled = (
            generated_starts and game_class in ("1Pop3S", "2Pop2S")
            if show_edge_flow is None
            else show_edge_flow
        )
        if edge_flow_enabled:
            if game_class not in ("1Pop3S", "2Pop2S"):
                raise ValueError("show_edge_flow=True is supported only for 1Pop3S and 2Pop2S.")
            edge_color = (
                trajectory_color
                if mcolors.is_color_like(trajectory_color)
                else trajectory_color[0]
            )
            _plot_edge_flow(
                game_class=game_class,
                payoff_data=payoff_data,
                fig=fig,
                ax=ax,
                show_arrows=not (
                    edge_arrow_positions is not None
                    and len(edge_arrow_positions) == 0
                ),
                trajectory_color=edge_color,
                trajectory_linewidth=trajectory_linewidth,
                arrow_size=arrow_size,
                arrow_width=arrow_width,
                zorder=trajectory_zorder,
            )

    if title:
        ax.set_title(title, pad=title_pad)

    return fig, ax


def _payoff_data(game):
    return getattr(game, "payoff_data", game)


def _strategy_labels(game, game_class):
    labels = getattr(game, "strategy_labels", None)
    if game_class == "3Pop2S":
        # Three-population cubes need one label per axis, whereas shared
        # strategy labels contain only the two strategies. Preserve explicit
        # three-axis labels used by catalogue games; otherwise construct the
        # labels from each population and its first listed strategy.
        if labels and len(labels) == 3:
            return labels
        player_strategy_labels = getattr(game, "player_strategy_labels", None)
        player_labels = getattr(game, "player_labels", None)
        if (
            player_strategy_labels
            and len(player_strategy_labels) == 3
            and all(strategy_set for strategy_set in player_strategy_labels)
        ):
            if not player_labels or len(player_labels) != 3:
                player_labels = ["Population 1", "Population 2", "Population 3"]
            return [
                f"{player_label}: Pr({strategy_labels[0]})"
                for player_label, strategy_labels in zip(
                    player_labels, player_strategy_labels
                )
            ]
        return ["Population 1", "Population 2", "Population 3"]
    if labels:
        return labels
    if game_class == "2Pop2S":
        return ["Strategy 1", "Strategy 2"]
    if game_class == "1Pop4S":
        return ["Strategy 1", "Strategy 2", "Strategy 3", "Strategy 4"]
    return ["Strategy 1", "Strategy 2", "Strategy 3"]


def _player_strategy_labels(game, game_class):
    labels = getattr(game, "player_strategy_labels", None)
    if labels:
        return labels
    shared = _strategy_labels(game, game_class)
    if game_class == "2Pop2S":
        return [shared, shared]
    if game_class == "3Pop2S":
        return [[label] for label in shared]
    return [shared]


def _player_labels(game, game_class):
    labels = getattr(game, "player_labels", None)
    if labels:
        return labels
    if game_class == "2Pop2S":
        return ["Population 1", "Population 2"]
    if game_class == "3Pop2S":
        return ["Population 1", "Population 2", "Population 3"]
    return ["Population"]


def _trajectory_colors(trajectory_color, count):
    if mcolors.is_color_like(trajectory_color):
        return [trajectory_color] * count
    if len(trajectory_color) != count:
        raise ValueError(
            "trajectory_color must be a single Matplotlib color or a list of "
            "colors with the same length as starts."
        )
    if not all(mcolors.is_color_like(color) for color in trajectory_color):
        raise ValueError("Every entry in trajectory_color must be a Matplotlib color.")
    return trajectory_color


def _apply_axis_label_overrides(ax, xlabel, ylabel, zlabel):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None and hasattr(ax, "set_zlabel"):
        ax.set_zlabel(zlabel)


def _get_or_create_axes(fig, ax, game_class, figsize, view_elev, view_azim):
    if ax is not None:
        return ax.figure, ax

    if fig is None:
        fig = plt.figure(figsize=figsize)

    if game_class in ("1Pop4S", "3Pop2S"):
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=view_elev, azim=view_azim)
    else:
        ax = fig.add_subplot(111)
        ax.set_aspect(1)

    return fig, ax


def _set_default_axes_style(ax, game_class, labels, player_strategy_labels, player_labels):
    if game_class == "2Pop2S":
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"{player_labels[0]}: Pr({player_strategy_labels[0][0]})")
        ax.set_ylabel(f"{player_labels[1]}: Pr({player_strategy_labels[1][0]})")
    elif game_class == "1Pop3S":
        ax.axis("off")
    elif game_class == "1Pop4S":
        ax.set_axis_off()


def _speed_regions(game_class):
    if game_class == "1Pop3S":
        return [0, 1], [0, np.sqrt(3 / 4)]
    return [0, 1], [0, 1]


def _default_starts(game_class, random_state, trajectory_number=4):
    if (
        not isinstance(trajectory_number, int)
        or isinstance(trajectory_number, bool)
        or trajectory_number < 1
    ):
        raise ValueError("trajectory_number must be a positive integer.")

    if random_state is not None:
        rng = np.random.default_rng(random_state)
        if game_class == "1Pop3S":
            return rng.dirichlet(np.ones(3), size=trajectory_number)[:, :2].tolist()
        if game_class == "1Pop4S":
            return rng.dirichlet(np.ones(4), size=trajectory_number)[:, :3].tolist()
        if game_class in ("2Pop2S", "3Pop2S"):
            dimension = 2 if game_class == "2Pop2S" else 3
            return rng.uniform(0.05, 0.95, size=(trajectory_number, dimension)).tolist()

    if game_class == "1Pop3S":
        return _simplex_partition_centroids(3, trajectory_number)[:, :2].tolist()
    if game_class == "1Pop4S":
        return _simplex_partition_centroids(4, trajectory_number)[:, :3].tolist()
    if game_class == "2Pop2S":
        return _box_partition_centroids(2, trajectory_number).tolist()
    if game_class == "3Pop2S":
        return _box_partition_centroids(3, trajectory_number).tolist()
    raise ValueError(f"Unsupported game class: {game_class}")


def _simplex_partition_centroids(strategy_count, count):
    """Return centroids of a deterministic recursive simplex partition."""
    cells = [np.eye(strategy_count)]

    def volume_measure(vertices):
        edges = vertices[1:] - vertices[0]
        return np.sqrt(max(np.linalg.det(edges @ edges.T), 0.0))

    while len(cells) < count:
        cell_index = max(range(len(cells)), key=lambda index: volume_measure(cells[index]))
        vertices = cells.pop(cell_index)
        pairs = [
            (i, j)
            for i in range(strategy_count)
            for j in range(i + 1, strategy_count)
        ]
        first, second = max(
            pairs,
            key=lambda pair: np.linalg.norm(vertices[pair[0]] - vertices[pair[1]]),
        )
        midpoint = (vertices[first] + vertices[second]) / 2
        child_1 = vertices.copy()
        child_2 = vertices.copy()
        child_1[first] = midpoint
        child_2[second] = midpoint
        cells.extend([child_1, child_2])

    return np.asarray([cell.mean(axis=0) for cell in cells])


def _box_partition_centroids(dimension, count):
    """Return centroids of a deterministic recursive unit-box partition."""
    boxes = [(np.zeros(dimension), np.ones(dimension))]
    while len(boxes) < count:
        box_index = max(
            range(len(boxes)),
            key=lambda index: np.prod(boxes[index][1] - boxes[index][0]),
        )
        lower, upper = boxes.pop(box_index)
        split_dimension = int(np.argmax(upper - lower))
        midpoint = (lower[split_dimension] + upper[split_dimension]) / 2
        upper_1 = upper.copy()
        upper_1[split_dimension] = midpoint
        lower_2 = lower.copy()
        lower_2[split_dimension] = midpoint
        boxes.extend([(lower, upper_1), (lower_2, upper)])

    return np.asarray([(lower + upper) / 2 for lower, upper in boxes])


def _vector_field_2p2s(payoff_data, grid, margin):
    values = np.linspace(margin, 1.0 - margin, grid)
    points = []
    vectors = []
    for x_val in values:
        for y_val in values:
            point = np.array([x_val, y_val])
            vector = np.asarray(dynamics.replicator_2p2s(point, 0, payoff_data), dtype=float)
            points.append(point)
            vectors.append(vector)
    return np.asarray(points), np.asarray(vectors)


def _vector_field_2p3s(payoff_data, grid, margin):
    bary_points = _simplex_grid_3(grid, margin)
    points = []
    vectors = []
    for r, p, _ in bary_points:
        vector = np.asarray(dynamics.replicator_2p3s([r, p], 0, payoff_data), dtype=float)
        start = np.asarray(simplex_to_plane_2p3s(r, p), dtype=float)
        end = np.asarray(simplex_to_plane_2p3s(r + vector[0], p + vector[1]), dtype=float)
        points.append(start)
        vectors.append(end - start)
    return np.asarray(points), np.asarray(vectors)


def _vector_field_3p2s(payoff_data, grid, margin):
    values = np.linspace(margin, 1.0 - margin, grid)
    points = []
    vectors = []
    for x_val in values:
        for y_val in values:
            for z_val in values:
                point = np.array([x_val, y_val, z_val])
                vector = np.asarray(dynamics.replicator_3p2s(point, 0, payoff_data), dtype=float)
                points.append(point)
                vectors.append(vector)
    return np.asarray(points), np.asarray(vectors)


def _vector_field_2p4s(payoff_data, grid, margin):
    bary_points = _simplex_grid_4(grid, margin)
    points = []
    vectors = []
    for a, b, c, _ in bary_points:
        vector = np.asarray(dynamics.replicator_2p4s([a, b, c], 0, payoff_data), dtype=float)
        start = np.asarray(simplex_to_plane_2p4s(a, b, c), dtype=float)
        end = np.asarray(simplex_to_plane_2p4s(a + vector[0], b + vector[1], c + vector[2]), dtype=float)
        points.append(start)
        vectors.append(end - start)
    return np.asarray(points), np.asarray(vectors)


def _simplex_grid_3(grid, margin):
    points = []
    values = np.linspace(margin, 1.0 - margin, grid)
    for a in values:
        for b in values:
            c = 1.0 - a - b
            if c >= margin:
                points.append([a, b, c])
    return np.asarray(points)


def _simplex_grid_4(grid, margin):
    points = []
    values = np.linspace(margin, 1.0 - margin, grid)
    for a in values:
        for b in values:
            for c in values:
                d = 1.0 - a - b - c
                if d >= margin:
                    points.append([a, b, c, d])
    return np.asarray(points)


def _normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    nonzero = norms > 1e-12
    normalized = np.zeros_like(vectors)
    normalized[nonzero] = vectors[nonzero] / norms[nonzero, None]
    return normalized, nonzero


def _quiver_2d(ax, points, vectors, color, alpha, length, width, zorder, normalize):
    if normalize:
        vectors, keep = _normalize_vectors(vectors)
        points = points[keep]
        vectors = vectors[keep]
        vectors = vectors * length
    if len(points) == 0:
        return None
    return ax.quiver(
        points[:, 0],
        points[:, 1],
        vectors[:, 0],
        vectors[:, 1],
        color=color,
        alpha=alpha,
        width=width,
        zorder=zorder,
        angles="xy",
        scale_units="xy",
        scale=1,
    )


def _quiver_3d(ax, points, vectors, color, alpha, length, zorder, normalize):
    if normalize:
        vectors, keep = _normalize_vectors(vectors)
        points = points[keep]
        vectors = vectors[keep]
    if len(points) == 0:
        return None
    return ax.quiver(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        vectors[:, 0],
        vectors[:, 1],
        vectors[:, 2],
        color=color,
        alpha=alpha,
        length=length,
        normalize=False,
        zorder=zorder,
    )


def _matrix_to_colors(matrix, cmap):
    """Converts a matrix into a RGBA color map."""
    color_dimension = matrix # It must be in 2D - as for "X, Y, Z".
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)
    return fcolors, m


def _outside_2p3s_simplex(X, Y):
    """Boolean mask for points lying outside the 1Pop3S simplex in plotting coordinates."""
    mask = np.zeros(X.shape, dtype=bool)
    for i in range(len(X)):
        for j in range(len(X)):
            if X[i, j] <= 0.5:
                mask[i, j] = Y[i, j] > 2 * np.sqrt(3 / 4) * X[i, j]
            elif X[i, j] > 0.5:
                mask[i, j] = Y[i, j] > 2 * np.sqrt(3 / 4) * (1 - X[i, j])
            else:
                mask[i, j] = False
    return mask


def _project_to_2p3s_simplex(X, Y):
    """Orthogonally project out-of-bounds grid points back to the 1Pop3S simplex."""
    mask = _outside_2p3s_simplex(X, Y)
    for i in range(len(X)):
        for j in range(len(Y)):
            if mask[i][j]:
                if X[i, j] < 0.5:
                    xB, yB = 0, 0
                    xV, yV = 1, 2 * np.sqrt(3 / 4)
                    BH = ((X[i, j] - xB) * xV + (Y[i, j] - yB) * yV) / (np.sqrt(xV**2 + yV**2))
                    X[i, j] = xB + (BH / np.sqrt(xV**2 + yV**2)) * xV
                    Y[i, j] = 2 * np.sqrt(3 / 4) * X[i, j]
                elif X[i, j] > 0.5:
                    xB, yB = 0.5, np.sqrt(3 / 4)
                    xV, yV = 1, -2 * np.sqrt(3 / 4)
                    BH = ((X[i, j] - xB) * xV + (Y[i, j] - yB) * yV) / (np.sqrt(xV**2 + yV**2))
                    X[i, j] = xB + (BH / np.sqrt(xV**2 + yV**2)) * xV
                    Y[i, j] = 2 * np.sqrt(3 / 4) * (1 - X[i, j])
    return X, Y


def _speed_2p3s(x, y, payoff_data):
    """Speed magnitude of the 1Pop3S replicator dynamics at plotting coordinates (x, y)."""
    r, p = plane_to_simplex_2p3s(x, y)
    vector = np.asarray(dynamics.replicator_2p3s([r, p], 0, payoff_data), dtype=float)
    start = np.asarray(simplex_to_plane_2p3s(r, p), dtype=float)
    end = np.asarray(simplex_to_plane_2p3s(r + vector[0], p + vector[1]), dtype=float)
    return np.linalg.norm(end - start)


def _speed_grid_2p3s(X, Y, payoff_data):
    """Fill a grid with speeds for 1Pop3S replicator dynamics."""
    CALC = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(Y)):
            CALC[i][j] = _speed_2p3s(X[i][j], Y[i][j], payoff_data)
    return CALC


def _speed_grid_2p2s(U, V, payoff_data):
    """Fill a grid with speeds for 2Pop2S replicator dynamics."""
    CALC = np.zeros(U.shape)
    for i in range(len(U)):
        for j in range(len(V)):
            x = U[i][j]
            y = V[i][j]
            vector = dynamics.replicator_2p2s([x, y], 0, payoff_data)
            CALC[i][j] = np.linalg.norm(vector)
    return CALC


def plot_speed_field(x_region, y_region, step, payoff_data, ax, cmap, levels, zorder):
    """Plot the speed magnitude of 2D replicator dynamics.

    Parameters
    ----------
    x_region, y_region : sequence of float
        Lower and upper plotting bounds for the grid.
    step : int
        Grid density used to evaluate speed.
    payoff_data : numpy.ndarray or tuple of numpy.ndarray
        Payoff representation of a 2Pop2S or 1Pop3S game.
    ax : matplotlib.axes.Axes
        Axes on which the speed field is drawn.
    cmap : matplotlib colormap
        Colormap used for the filled contour plot.
    levels : int
        Number of contour levels.
    zorder : float
        Drawing order of the speed field.

    Returns
    -------
    matplotlib.contour.QuadContourSet or None
        Filled contour artist for supported 2D games; None otherwise.
    """
    game_class = infer_game_class(payoff_data)
    x = np.linspace(x_region[0], x_region[1], step)
    y = np.linspace(y_region[0], y_region[1], step)
    X, Y = np.meshgrid(x, y)

    if game_class == "1Pop3S":
        X, Y = _project_to_2p3s_simplex(X, Y)
        C = _speed_grid_2p3s(X, Y, payoff_data)
        surf = ax.contourf(
            X,
            Y,
            C,
            levels=levels,
            cmap=cmap,
            corner_mask=False,
            alpha=0.9,
            zorder=zorder,
        )
        return surf

    if game_class == "2Pop2S":
        C = _speed_grid_2p2s(X, Y, payoff_data)
        surf = ax.contourf(
            X,
            Y,
            C,
            levels=levels,
            cmap=cmap,
            corner_mask=False,
            alpha=0.9,
            zorder=zorder,
        )
        return surf

    raise NotImplementedError("Speed plot currently supports only 1Pop3S and 2Pop2S games.")


# Imported after plot defaults are defined; the shared renderer uses them.
from ._one_dimensional import bifurcation_diagram
