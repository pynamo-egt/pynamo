"""Backend-neutral description of a phase portrait.

A scene is a plain, JSON-serialisable dict. Renderers (Matplotlib now, Plotly
later) draw it; nothing in a scene depends on a plotting library.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.integrate import odeint

from . import analysis, dynamics
from .game import require_supported_game_class
from .state_space import StateSpace

__all__ = ["SCENE_VERSION", "scene"]

SCENE_VERSION = 1

# Vertex label placement relative to its vertex, per drawable shape.
_LABEL_SIDES = {"triangle": ["above", "below", "below"]}


def scene(
    game,
    dynamic: Optional[dynamics.Replicator] = None,
    *,
    starts: Optional[Sequence[Sequence[float]]] = None,
    trajectory_number: int = 4,
    random_state: Optional[int] = None,
    tmax: float = 45,
    trajectory_step: float = 0.02,
    trajectory_arrows: Sequence[float] = (0.02,),
    show_edge_flow: Optional[bool] = None,
) -> dict[str, Any]:
    """Describe the phase portrait of ``game`` under ``dynamic``.

    Parameters mirror :func:`pynamo_egt.phase_portrait`. Currently only
    one-population three-strategy games (1Pop3S) are supported.

    Parameters
    ----------
    game : Game or payoff data
        Game to draw.
    dynamic : Replicator, optional
        Dynamic to draw; the replicator by default.
    starts : sequence of sequence of float, optional
        Reduced initial states. By default, ``trajectory_number`` states from
        :meth:`StateSpace.default_starts`.
    trajectory_number, random_state : optional
        Number and seed of default starts.
    tmax, trajectory_step : float
        Integration horizon and step, forward and backward from each start.
    trajectory_arrows : sequence of float
        Fractions of each forward trajectory where a direction arrow sits.
    show_edge_flow : bool, optional
        Draw the flow along the edges. By default, only with default starts and
        only when the dynamic leaves the boundary invariant.

    Returns
    -------
    dict
        JSON-serialisable scene, deterministic for given inputs.
    """
    dynamic = dynamics.Replicator() if dynamic is None else dynamic
    game_class = require_supported_game_class(game)
    space = StateSpace.for_game(game)
    if space.kind != "triangle":
        raise NotImplementedError(f"Scenes for {game_class} games are not available yet.")

    generated = starts is None
    initial_states = space.default_starts(trajectory_number, random_state) if starts is None else starts
    if show_edge_flow is None:
        show_edge_flow = generated and dynamic.boundary_invariant

    return {
        "version": SCENE_VERSION,
        "game_class": game_class,
        "title": getattr(game, "name", "") or "",
        "dynamic": dynamic.name,
        "state_space": {"kind": space.kind, "populations": list(space.populations)},
        "frame": _frame(game, space),
        "rest_points": _rest_points(game, space),
        "trajectories": [
            _trajectory(game, dynamic, space, start, tmax, trajectory_step, trajectory_arrows)
            for start in initial_states
        ],
        "edge_flow": _edge_flow(game, space) if show_edge_flow else [],
    }


def _frame(game, space: StateSpace) -> dict[str, Any]:
    vertices = space.embed(np.vstack([np.eye(2), np.zeros(2)]))
    labels = list(getattr(game, "strategy_labels", None) or ["Strategy 1", "Strategy 2", "Strategy 3"])
    return {
        "edges": [_points(vertices[[i, j]]) for i, j in ((0, 1), (0, 2), (1, 2))],
        "vertex_labels": [
            {"text": text, "position": _points(vertex), "side": side}
            for text, vertex, side in zip(labels, vertices, _LABEL_SIDES[space.kind])
        ],
    }


def _rest_points(game, space: StateSpace) -> list[dict[str, Any]]:
    result = analysis.analyze_equilibria(game)
    return [
        {
            "position": _points(point.full_position),
            "plot": _points(space.embed(point.reduced_position)),
            "stability": point.stability,
        }
        for point in result.equilibria
    ]


def _trajectory(game, dynamic, space, start, tmax, step, arrow_fractions) -> dict[str, Any]:
    times = np.linspace(0, tmax, int(tmax / step))
    paths = {
        direction: space.embed(
            odeint(lambda state, _t, sign=sign: sign * dynamic.velocity(game, state), start, times)
        )
        for direction, sign in (("forward", 1.0), ("backward", -1.0))
    }
    forward = paths["forward"]
    arrows = []
    for fraction in arrow_fractions:
        base = min(max(int(fraction * (len(forward) - 1)), 0), len(forward) - 2)
        arrows.append(_points(forward[base:base + 2]))
    return {
        "start": _points(start),
        "forward": _points(forward),
        "backward": _points(paths["backward"]),
        "arrows": arrows,
    }


def _edge_flow(game, space: StateSpace) -> list[dict[str, Any]]:
    """Exact one-dimensional flow along each edge, with arrows between rest points."""
    matrix = np.asarray(getattr(game, "payoff_data", game), dtype=float)
    edges = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        edge_matrix = matrix[np.ix_([first, second], [first, second])]

        def embed(frequency: float, first=first, second=second) -> np.ndarray:
            state = np.zeros(3)
            state[first], state[second] = frequency, 1 - frequency
            return space.embed(state[:2])

        points, continuum = dynamics._analyze_1d(edge_matrix)
        arrows = []
        if not continuum:
            for left, right in zip(points[:-1], points[1:]):
                middle = (left.x + right.x) / 2
                direction = np.sign(dynamics.replicator_1d(middle, edge_matrix))
                if direction:
                    arrows.append(_points(np.stack([embed(middle - direction * 0.01), embed(middle)])))
        edges.append({"segment": _points(np.stack([embed(0.0), embed(1.0)])), "arrows": arrows})
    return edges


def _points(values) -> Any:
    """Nested lists of Python floats."""
    return np.asarray(values, dtype=float).tolist()
