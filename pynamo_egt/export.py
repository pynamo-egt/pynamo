"""Plotly-based interactive HTML export for pyNamo phase portraits.

Supports 1Pop4S (tetrahedron) and 3Pop2S (cube) game classes.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from . import analysis, dynamics
from .drawer import simplex_to_plane_2p4s
from .game import infer_game_class

__all__ = ["phase_portrait_html"]

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
    _PLOTLY_IMPORT_ERROR = None
except ImportError as _exc:
    _PLOTLY_AVAILABLE = False
    _PLOTLY_IMPORT_ERROR = _exc

_SUPPORTED_CLASSES = ("1Pop4S", "3Pop2S")


def phase_portrait_html(
    game,
    path=None,
    *,
    starts=None,
    tmax=45,
    trajectory_step=0.05,
    trajectory_number=4,
    trajectory_color="royalblue",
    trajectory_arrows=None,
    show_equilibria=True,
    sink_color="black",
    source_color="white",
    saddle_color="gray",
    show_speed=True,
    show_faces=True,
    face_alpha=0.12,
    random_state=None,
):
    """Export an interactive 3D phase portrait as a self-contained HTML file.

    Parameters
    ----------
    game : Game or payoff data
        A 1Pop4S or 3Pop2S game. Raises ValueError for other game classes.
    path : str or None
        If given, write the HTML to this file.
    starts : list or None
        Initial conditions in reduced coordinates. If None,
        ``trajectory_number`` starts are sampled from a uniform/Dirichlet
        distribution.
    tmax : float
        Forward and backward integration time.
    trajectory_step : float
        ODE integration step size.
    trajectory_number : int
        Number of trajectories when ``starts`` is None.
    trajectory_color : str or list of str
        Plotly-compatible color(s) for trajectory lines and arrow cones.
    trajectory_arrows : list of float or None
        Fractions along the forward trajectory at which to draw directional
        cones. Default is ``[0.02]``.
    show_equilibria : bool
        Whether to draw equilibrium markers.
    sink_color, source_color, saddle_color : str
        Colors for the respective stability types.
    show_speed : bool
        Render each face as a speed-coloured mesh (Spectral_r: blue = slow,
        red = fast). When True, ``show_faces`` is ignored.
    show_faces : bool
        When ``show_speed`` is False, whether to draw semi-transparent grey
        face panels.
    face_alpha : float
        Opacity of plain grey face panels.
    random_state : int or None
        Seed for random start generation.

    Returns
    -------
    str
        Self-contained HTML string with Plotly loaded from CDN.

    Raises
    ------
    ImportError
        If plotly is not installed.
    ValueError
        If the game class is not 1Pop4S or 3Pop2S.
    """
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for phase_portrait_html. "
            'Install it with: pip install "pynamo-egt[plotly]"'
        ) from _PLOTLY_IMPORT_ERROR

    payoff_data = getattr(game, "payoff_data", game)
    game_class = infer_game_class(payoff_data)
    if game_class not in _SUPPORTED_CLASSES:
        raise ValueError(
            f"phase_portrait_html supports {_SUPPORTED_CLASSES}; got '{game_class}'."
        )

    game_name = getattr(game, "name", "") or ""
    strategy_labels = _get_strategy_labels(game, game_class)

    if trajectory_arrows is None:
        trajectory_arrows = [0.02]

    traces = []

    # ------------------------------------------------------------------ #
    # State-space geometry                                                 #
    # ------------------------------------------------------------------ #
    if game_class == "1Pop4S":
        traces.extend(_1pop4s_geometry(strategy_labels, show_speed, show_faces, face_alpha, payoff_data))
    else:
        traces.extend(_3pop2s_geometry(show_speed, show_faces, face_alpha, payoff_data))

    # ------------------------------------------------------------------ #
    # Trajectories                                                         #
    # ------------------------------------------------------------------ #
    t = np.linspace(0, tmax, max(int(tmax / trajectory_step), 2))

    if starts is None:
        rng = np.random.default_rng(random_state)
        if game_class == "1Pop4S":
            raw = rng.dirichlet(np.ones(4), size=trajectory_number)
            starts = raw[:, :3].tolist()
        else:
            starts = rng.uniform(0.05, 0.95, (trajectory_number, 3)).tolist()

    if isinstance(trajectory_color, str):
        colors = [trajectory_color] * len(starts)
    else:
        colors = list(trajectory_color)
        colors += [colors[-1]] * max(0, len(starts) - len(colors))

    fwd_field = dynamics.replicator_2p4s if game_class == "1Pop4S" else dynamics.replicator_3p2s
    bwd_field = dynamics.reverse_replicator_2p4s if game_class == "1Pop4S" else dynamics.reverse_replicator_3p2s

    for start, color in zip(starts, colors):
        fwd = odeint(fwd_field, list(start), t, (payoff_data,))
        bwd = odeint(bwd_field, list(start), t, (payoff_data,))

        fwd_xyz = _to_xyz(fwd, game_class)
        bwd_xyz = _to_xyz(bwd, game_class)

        for xyz in (fwd_xyz, bwd_xyz):
            traces.append(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="lines",
                line=dict(color=color, width=3),
                showlegend=False,
                hoverinfo="skip",
            ))

        for frac in trajectory_arrows:
            idx = min(max(int(frac * (len(fwd_xyz) - 1)), 0), len(fwd_xyz) - 2)
            direction = fwd_xyz[idx + 1] - fwd_xyz[idx]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction /= norm
            pos = fwd_xyz[idx]
            traces.append(go.Cone(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                u=[direction[0]], v=[direction[1]], w=[direction[2]],
                sizemode="absolute",
                sizeref=0.04,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                anchor="tip",
                hoverinfo="skip",
            ))

    # ------------------------------------------------------------------ #
    # Equilibria                                                           #
    # ------------------------------------------------------------------ #
    if show_equilibria:
        _stability_colors = {
            "sink": sink_color,
            "source": source_color,
            "saddle": saddle_color,
            "unstable": source_color,
            "center": source_color,
            "undetermined": "lightgrey",
        }
        result = analysis.analyze_equilibria(payoff_data)
        by_stability: dict = {}
        for eq in result.equilibria:
            pt = _eq_to_xyz(eq, game_class)
            by_stability.setdefault(eq.stability, []).append(pt)

        for stab, pts in by_stability.items():
            pts_arr = np.array(pts)
            color = _stability_colors.get(stab, "grey")
            traces.append(go.Scatter3d(
                x=pts_arr[:, 0], y=pts_arr[:, 1], z=pts_arr[:, 2],
                mode="markers",
                marker=dict(size=8, color=color, line=dict(color="black", width=1)),
                name=stab,
                showlegend=True,
            ))

    # ------------------------------------------------------------------ #
    # Layout                                                               #
    # ------------------------------------------------------------------ #
    if game_class == "1Pop4S":
        scene = dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="white",
            aspectmode="data",
        )
    else:
        scene = dict(
            xaxis=dict(title=strategy_labels[0], range=[0, 1]),
            yaxis=dict(title=strategy_labels[1], range=[0, 1]),
            zaxis=dict(title=strategy_labels[2], range=[0, 1]),
            bgcolor="white",
            aspectmode="cube",
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=game_name,
        scene=scene,
        margin=dict(l=0, r=0, t=40 if game_name else 10, b=0),
        paper_bgcolor="white",
        legend=dict(title="Stability"),
    )

    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)

    if path is not None:
        with open(path, "w") as f:
            f.write(html_str)

    return html_str


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def _1pop4s_geometry(strategy_labels, show_speed, show_faces, face_alpha, payoff_data):
    v = [
        np.array(simplex_to_plane_2p4s(1, 0, 0)),
        np.array(simplex_to_plane_2p4s(0, 1, 0)),
        np.array(simplex_to_plane_2p4s(0, 0, 1)),
        np.array(simplex_to_plane_2p4s(0, 0, 0)),
    ]
    face_vertex_indices = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    traces = []

    if show_speed:
        traces.extend(_1pop4s_speed_faces(payoff_data))
    elif show_faces:
        for a, b, c in face_vertex_indices:
            traces.append(go.Mesh3d(
                x=[v[a][0], v[b][0], v[c][0]],
                y=[v[a][1], v[b][1], v[c][1]],
                z=[v[a][2], v[b][2], v[c][2]],
                i=[0], j=[1], k=[2],
                color="lightgrey",
                opacity=face_alpha,
                showlegend=False,
                hoverinfo="skip",
            ))

    for a, b in [(0,1),(1,2),(2,0),(3,0),(3,1),(3,2)]:
        traces.append(go.Scatter3d(
            x=[v[a][0], v[b][0], None],
            y=[v[a][1], v[b][1], None],
            z=[v[a][2], v[b][2], None],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    offsets = [
        np.array([0.0,  0.06, 0.0]),
        np.array([-0.08, -0.04, 0.0]),
        np.array([0.08, -0.04, 0.0]),
        np.array([0.0,  0.0,  0.07]),
    ]
    for vertex, label, offset in zip(v, strategy_labels, offsets):
        pos = vertex + offset
        traces.append(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="text",
            text=[label],
            textfont=dict(size=14, color="black"),
            showlegend=False,
            hoverinfo="skip",
        ))

    return traces


def _3pop2s_geometry(show_speed, show_faces, face_alpha, payoff_data):
    corners = np.array([
        [0,0,0],[1,0,0],[0,1,0],[1,1,0],
        [0,0,1],[1,0,1],[0,1,1],[1,1,1],
    ], dtype=float)
    edges = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)]
    traces = []

    if show_speed:
        traces.extend(_3pop2s_speed_faces(payoff_data))
    elif show_faces:
        for fixed_axis, fixed_value in [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]:
            pts, tris = _square_face_samples(fixed_axis, fixed_value, n_grid=2)
            traces.append(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                color="lightgrey",
                opacity=face_alpha,
                showlegend=False,
                hoverinfo="skip",
            ))

    for a, b in edges:
        traces.append(go.Scatter3d(
            x=[corners[a][0], corners[b][0], None],
            y=[corners[a][1], corners[b][1], None],
            z=[corners[a][2], corners[b][2], None],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    return traces


# ---------------------------------------------------------------------------
# Speed-coloured face meshes
# ---------------------------------------------------------------------------

def _1pop4s_speed_faces(payoff_data, n_grid=20):
    tris = _triangular_grid_indices(n_grid)
    traces = []
    for face_index in range(4):
        simplex_pts, xyz_pts = _simplex_face_samples(face_index, n_grid)
        speeds = np.array([
            np.linalg.norm(dynamics.replicator_2p4s(pt, 0, payoff_data))
            for pt in simplex_pts
        ])
        show_bar = face_index == 0
        traces.append(go.Mesh3d(
            x=xyz_pts[:, 0], y=xyz_pts[:, 1], z=xyz_pts[:, 2],
            i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
            intensity=speeds,
            intensitymode="vertex",
            colorscale="Spectral",
            reversescale=True,
            showscale=show_bar,
            colorbar=dict(title="Speed", thickness=12, len=0.5) if show_bar else None,
            opacity=0.75,
            showlegend=False,
            hoverinfo="skip",
        ))
    return traces


def _3pop2s_speed_faces(payoff_data, n_grid=20):
    traces = []
    first = True
    for fixed_axis in range(3):
        for fixed_value in (0.0, 1.0):
            pts, tris = _square_face_samples(fixed_axis, fixed_value, n_grid)
            speeds = np.array([
                np.linalg.norm(dynamics.replicator_3p2s(pt, 0, payoff_data))
                for pt in pts
            ])
            traces.append(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                intensity=speeds,
                intensitymode="vertex",
                colorscale="Spectral",
                reversescale=True,
                showscale=first,
                colorbar=dict(title="Speed", thickness=12, len=0.5) if first else None,
                opacity=0.75,
                showlegend=False,
                hoverinfo="skip",
            ))
            first = False
    return traces


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _to_xyz(path, game_class):
    """Convert an ODE solution array to 3-D plotting coordinates."""
    if game_class == "1Pop4S":
        return np.array([simplex_to_plane_2p4s(p[0], p[1], p[2]) for p in path])
    return np.array(path, dtype=float)


def _eq_to_xyz(eq, game_class):
    """Convert an AnalyzedEquilibrium to a 3-D plotting point."""
    r = eq.reduced_position
    if game_class == "1Pop4S":
        return np.array(simplex_to_plane_2p4s(r[0], r[1], r[2]))
    return np.array(r, dtype=float)


def _get_strategy_labels(game, game_class):
    labels = getattr(game, "strategy_labels", None)
    if game_class == "3Pop2S":
        if labels and len(labels) == 3:
            return list(labels)
        psl = getattr(game, "player_strategy_labels", None)
        pl = getattr(game, "player_labels", None)
        if psl and len(psl) == 3 and all(psl):
            if not pl or len(pl) != 3:
                pl = ["Pop. 1", "Pop. 2", "Pop. 3"]
            return [f"{p}: Pr({s[0]})" for p, s in zip(pl, psl)]
        return ["Pop. 1", "Pop. 2", "Pop. 3"]
    if labels:
        return list(labels)
    return [f"S{i + 1}" for i in range(4)]


# ---------------------------------------------------------------------------
# Triangulation helpers
# ---------------------------------------------------------------------------

def _simplex_face_samples(face_index, n_grid):
    """Triangular grid of (n_grid+1)(n_grid+2)/2 points on a simplex face."""
    other = [j for j in range(4) if j != face_index]
    simplex_pts, xyz_pts = [], []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            k = n_grid - i - j
            a, b, c = i / n_grid, j / n_grid, k / n_grid
            coords = [0.0, 0.0, 0.0, 0.0]
            coords[other[0]], coords[other[1]], coords[other[2]] = a, b, c
            simplex_pts.append(coords[:3])
            xyz_pts.append(simplex_to_plane_2p4s(coords[0], coords[1], coords[2]))
    return np.array(simplex_pts), np.array(xyz_pts)


def _triangular_grid_indices(n_grid):
    """Triangle index triples for a (n_grid+1)(n_grid+2)/2-point triangular grid."""
    pos = {}
    k = 0
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            pos[(i, j)] = k
            k += 1
    tris = []
    for i in range(n_grid):
        for j in range(n_grid - i):
            tris.append((pos[(i, j)], pos[(i + 1, j)], pos[(i, j + 1)]))
            if i + j + 2 <= n_grid:
                tris.append((pos[(i + 1, j)], pos[(i + 1, j + 1)], pos[(i, j + 1)]))
    return np.array(tris)


def _square_face_samples(fixed_axis, fixed_value, n_grid):
    """Regular (n_grid+1)² grid on a unit-square face of the cube."""
    u = np.linspace(0, 1, n_grid + 1)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    uu, vv = uu.ravel(), vv.ravel()
    free = [i for i in range(3) if i != fixed_axis]
    pts = np.zeros((len(uu), 3))
    pts[:, fixed_axis] = fixed_value
    pts[:, free[0]] = uu
    pts[:, free[1]] = vv

    n = n_grid + 1
    tris = []
    for i in range(n_grid):
        for j in range(n_grid):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            tris.append((a, b, c))
            tris.append((b, d, c))
    return pts, np.array(tris)
