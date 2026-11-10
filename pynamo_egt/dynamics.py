"""Replicator vector fields and rest-point computation for pyNamo games."""

from dataclasses import dataclass
from typing import Union
import warnings

import numpy as np

from . import _rest_points
from .game import PayoffCollection, PayoffMatrix, infer_game_class

__all__ = [
    "DegenerateEquilibriumWarning",
    "EquilibriumResult",
    "replicator_2p2s",
    "reverse_replicator_2p2s",
    "replicator_2p3s",
    "reverse_replicator_2p3s",
    "replicator_2p4s",
    "reverse_replicator_2p4s",
    "replicator_3p2s",
    "reverse_replicator_3p2s",
    "compute_equilibria",
    "replicator_1d",
]


class DegenerateEquilibriumWarning(RuntimeWarning):
    """Warning raised when equilibrium sets are non-isolated."""


# Kept verbatim from the SymPy-based solver so existing output is unchanged.
_DEGENERATE_MESSAGE = (
    "Degenerate equilibrium solutions detected: SymPy returned a "
    "parametric/non-isolated solution set. Non-isolated rest-point "
    "families are not expanded into points; vertices and identifiable "
    "isolated points are retained."
)


class EquilibriumResult(list):
    """List of isolated equilibria with metadata about non-isolated solutions."""

    def __init__(self, equilibria=None, *, degenerate=False, message=None):
        super().__init__(equilibria or [])
        self.degenerate = degenerate
        self.message = message

# Replicator dynamics for a symmetric 1Pop3S game
def _average_payoff_2p3s(x1, x2, y1, y2, payoff_data):
    """Average payoff of strategy (x1, x2) against strategy (y1, y2)."""
    return (
        x1 * (y1 * payoff_data[0][0] + y2 * payoff_data[0][1] + (1 - y1 - y2) * payoff_data[0][2])
        + x2 * (y1 * payoff_data[1][0] + y2 * payoff_data[1][1] + (1 - y1 - y2) * payoff_data[1][2])
        + (1 - x1 - x2)
        * (y1 * payoff_data[2][0] + y2 * payoff_data[2][1] + (1 - y1 - y2) * payoff_data[2][2])
    )


def replicator_2p3s(state, t, payoff_data):
    """Replicator dynamics for a symmetric 2-player 3-strategy game."""
    x1, x2 = state
    average_payoff = _average_payoff_2p3s(x1, x2, x1, x2, payoff_data)
    return np.array(
        [
            x1 * (_average_payoff_2p3s(1, 0, x1, x2, payoff_data) - average_payoff),
            x2 * (_average_payoff_2p3s(0, 1, x1, x2, payoff_data) - average_payoff),
        ]
    )


def reverse_replicator_2p3s(state, t, payoff_data):
    """Reverse-time replicator dynamics for a symmetric 1Pop3S game."""
    return -replicator_2p3s(state, t, payoff_data)


# Replicator dynamics for an asymmetric 2Pop2S game
def _replicator_2p2s_population(state, t, payoff_data):
    """One-population component of asymmetric 2Pop2S replicator dynamics."""
    x, y = state
    payoff_action_1 = y * payoff_data[0][0] + (1 - y) * payoff_data[0][1]
    payoff_action_2 = y * payoff_data[1][0] + (1 - y) * payoff_data[1][1]
    return x * (1 - x) * (payoff_action_1 - payoff_action_2)


def replicator_2p2s(state, t, payoff_data):
    """Vector field for asymmetric 2-player 2-strategy replicator dynamics.

    Each player's payoff matrix uses that player's own strategies as rows and
    the opponent's strategies as columns. The state coordinates are the
    probabilities of the players' respective first-row strategies.
    """
    x, y = state
    return np.array([
        _replicator_2p2s_population([x, y], t, payoff_data[0]),
        _replicator_2p2s_population([y, x], t, payoff_data[1]),
    ])


def reverse_replicator_2p2s(state, t, payoff_data):
    """Reverse-time vector field for asymmetric 2Pop2S replicator dynamics."""
    return -np.asarray(replicator_2p2s(state, t, payoff_data))

# Replicator dynamics for 1Pop4S game

def _average_payoff_2p4s(x1, x2, x3, y1, y2, y3, payoff_data):
    """Average payoff of strategy (x1, x2, x3) against strategy (y1, y2, y3)."""
    return x1*(y1*payoff_data[0, 0] + y2*payoff_data[0, 1] + y3*payoff_data[0, 2] + (1 - y1 - y2 - y3)*payoff_data[0, 3]) + x2*(y1*payoff_data[1, 0] + y2*payoff_data[1, 1] + y3*payoff_data[1, 2] + (1 - y1 - y2 - y3)*payoff_data[1, 3]) + x3*(y1*payoff_data[2, 0] + y2*payoff_data[2, 1] + y3*payoff_data[2, 2] + (1 - y1 - y2 - y3)*payoff_data[2, 3]) + (1 - x1 - x2 - x3)*(y1*payoff_data[3, 0] + y2*payoff_data[3, 1] + y3*payoff_data[3, 2] + (1 - y1 - y2 - y3)*payoff_data[3, 3])


def replicator_2p4s(state, t, payoff_data):
    """Replicator dynamics for a symmetric 2-player 4-strategy game."""
    x1, x2, x3 = state
    average_payoff = _average_payoff_2p4s(x1, x2, x3, x1, x2, x3, payoff_data)
    return np.array([x1*(_average_payoff_2p4s(1, 0, 0, x1, x2, x3, payoff_data) - average_payoff), x2*(_average_payoff_2p4s(0, 1, 0, x1, x2, x3, payoff_data) - average_payoff), x3*(_average_payoff_2p4s(0, 0, 1, x1, x2, x3, payoff_data) - average_payoff)])


def reverse_replicator_2p4s(state, t, payoff_data):
    """Reverse-time replicator dynamics for a symmetric 1Pop4S game."""
    return -replicator_2p4s(state, t, payoff_data)


# Replicator dynamics for a 3-player 2-strategy game
def _average_payoff_3p2s(pay_tensor, probs, player_index, action):
    """Compute expected payoff for a given player and action in a 3-player 2-strategy game."""
    total = 0.0
    for a0 in (0, 1):
        for a1 in (0, 1):
            for a2 in (0, 1):
                actions = [a0, a1, a2]
                if actions[player_index] != action:
                    continue
                prob = 1.0
                for idx, a_val in enumerate(actions):
                    if idx == player_index:
                        continue
                    p = probs[idx]
                    prob *= p if a_val == 0 else (1 - p)
                total += pay_tensor[a0, a1, a2] * prob
    return total


def replicator_3p2s(state, t, payoff_tensors):
    """Replicator dynamics for three players with two strategies each."""
    x, y, z = state
    probs = [x, y, z]
    tensors = payoff_tensors

    payoff_data = []
    for idx, tensor in enumerate(tensors):
        u0 = _average_payoff_3p2s(tensor, probs, idx, 0)
        u1 = _average_payoff_3p2s(tensor, probs, idx, 1)
        payoff_data.append((u0, u1))

    dx = x * (payoff_data[0][0] - (x * payoff_data[0][0] + (1 - x) * payoff_data[0][1]))
    dy = y * (payoff_data[1][0] - (y * payoff_data[1][0] + (1 - y) * payoff_data[1][1]))
    dz = z * (payoff_data[2][0] - (z * payoff_data[2][0] + (1 - z) * payoff_data[2][1]))
    return np.array([dx, dy, dz])


def reverse_replicator_3p2s(state, t, payoff_tensors):
    """Opposite replicator dynamics for the three-player 2-strategy game."""
    return -replicator_3p2s(state, t, payoff_tensors)


def compute_equilibria(payoff_data: Union[PayoffMatrix, PayoffCollection]) -> EquilibriumResult:
    """Return rest points (equilibria) of the replicator dynamics for the given payoff tensors."""
    game_class = infer_game_class(payoff_data)
    if game_class == "1Pop2S":
        points, continuum = _analyze_1d(payoff_data)
        return EquilibriumResult(
            [[p.x] for p in points], degenerate=continuum,
            message="Every frequency in [0, 1] is an equilibrium." if continuum else None,
        )
    if game_class in ("1Pop3S", "1Pop4S"):
        points, degenerate = _rest_points.symmetric_rest_points(payoff_data)
        message = None
        if degenerate:
            message = _DEGENERATE_MESSAGE
            warnings.warn(message, DegenerateEquilibriumWarning, stacklevel=2)
        return EquilibriumResult(
            [[float(value) for value in point[:-1]] for point in points],
            degenerate=degenerate,
            message=message,
        )

    from sympy import Symbol
    from sympy.solvers import solve
    from itertools import product

    time_0 = 0
    x_sym = Symbol('x')
    y_sym = Symbol('y')
    equilibria = []
    degenerate = False
    message = None

    def _split_solution_branches(solutions, symbols):
        isolated = []
        parametric = []
        found_degeneracy = False
        for sol in solutions:
            if not all(sym in sol for sym in symbols):
                found_degeneracy = True
                parametric.append(sol)
                continue
            try:
                isolated.append([float(sol[sym]) for sym in symbols])
            except (TypeError, ValueError):
                found_degeneracy = True
                parametric.append(sol)

        isolated = [
            point
            for point in isolated
            if _is_vertex(point) or not _point_in_parametric_branch(point, parametric, symbols)
        ]
        return isolated, parametric, found_degeneracy

    def _is_vertex(point):
        return all(abs(coord) < 1e-9 or abs(coord - 1.0) < 1e-9 for coord in point)

    def _point_in_parametric_branch(point, branches, symbols):
        substitutions = {
            sym: point[idx]
            for idx, sym in enumerate(symbols)
        }
        for branch in branches:
            contained = True
            for idx, (sym, expression) in enumerate(branch.items()):
                if sym not in symbols:
                    contained = False
                    break
                try:
                    value = float(expression.subs(substitutions))
                except AttributeError:
                    value = float(expression)
                except (TypeError, ValueError):
                    contained = False
                    break
                point_index = symbols.index(sym)
                if abs(point[point_index] - value) >= 1e-9:
                    contained = False
                    break
            if contained:
                return True
        return False

    def _mark_degenerate():
        nonlocal degenerate, message
        degenerate = True
        message = _DEGENERATE_MESSAGE
        warnings.warn(message, DegenerateEquilibriumWarning, stacklevel=3)

    def _contains_point(point, acc):
        for sol in acc:
            if all(abs(sol[i] - point[i]) < 1e-9 for i in range(len(point))):
                return True
        return False

    def _add_missing_points(points, parametric=(), symbols=()):
        for point in points:
            if (
                not _contains_point(point, equilibria)
                and (
                    _is_vertex(point)
                    or not _point_in_parametric_branch(point, parametric, symbols)
                )
            ):
                equilibria.append(list(point))

    if game_class == "2Pop2S":
        pay_p1, pay_p2 = payoff_data
        dx = _replicator_2p2s_population([x_sym, y_sym], time_0, pay_p1)
        dy = _replicator_2p2s_population([y_sym, x_sym], time_0, pay_p2)
        solutions = [{}] if dx == 0 and dy == 0 else solve([dx, dy], x_sym, y_sym, dict=True)
        symbols = (x_sym, y_sym)
        equilibria, parametric, degenerate = _split_solution_branches(solutions, symbols)
        if degenerate:
            _mark_degenerate()
        _add_missing_points(product((0.0, 1.0), repeat=2), parametric, symbols)

    elif game_class == "3Pop2S":
        z_sym = Symbol('z')
        dx, dy, dz = replicator_3p2s([x_sym, y_sym, z_sym], time_0, payoff_data)
        solutions = [{}] if dx == 0 and dy == 0 and dz == 0 else solve([dx, dy, dz], x_sym, y_sym, z_sym, dict=True)
        symbols = (x_sym, y_sym, z_sym)
        equilibria, parametric, degenerate = _split_solution_branches(solutions, symbols)
        if degenerate:
            _mark_degenerate()

        for vertex in product((0.0, 1.0), repeat=3):
            if (
                not _contains_point(vertex, equilibria)
                and (
                    _is_vertex(vertex)
                    or not _point_in_parametric_branch(vertex, parametric, symbols)
                )
            ):
                equilibria.append(list(vertex))

    return EquilibriumResult(equilibria, degenerate=degenerate, message=message)


@dataclass(frozen=True)
class _RestPoint1D:
    x: float
    stable: bool
    nonhyperbolic: bool = False


def _coefficients_1d(payoffs):
    A = np.asarray(payoffs, dtype=float)
    if A.shape != (2, 2) or not np.isfinite(A).all():
        raise ValueError("Expected a finite 2×2 payoff matrix.")
    a, b, c, d = A.ravel()
    return b - d, a - b - c + d


def _analyze_1d(payoffs, atol=1e-12):
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative.")
    u, v = _coefficients_1d(payoffs)
    zero = lambda z: abs(z) <= atol
    if zero(u) and zero(v):
        return (), True
    left = v if zero(u) else u
    right = -v if zero(u + v) else u + v
    points = [_RestPoint1D(0., left < 0, zero(u)),
              _RestPoint1D(1., right > 0, zero(u + v))]
    if not zero(v):
        x = -u / v
        if 0 < x < 1 and not zero(u) and not zero(u + v):
            points.append(_RestPoint1D(float(x), v < 0))
    return tuple(sorted(points, key=lambda p: p.x)), False


def replicator_1d(x, payoffs):
    """Scalar/vector flow for a symmetric 2x2 game; x is first-strategy frequency.

    Unlike the multi-coordinate ODE callbacks, this autonomous helper accepts
    just x and the payoff matrix. Arrays of frequencies are supported.
    """
    x = np.asarray(x, dtype=float)
    u, v = _coefficients_1d(payoffs)
    return x * (1 - x) * (u + v * x)
