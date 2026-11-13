"""Structured equilibrium analysis for pyNamo games.

This module separates mathematical analysis from plotting. Plotting functions
can consume analyzed equilibria, while notebooks can display them as tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import warnings

import numpy as np
from sympy import Matrix
from sympy.abc import x, y, z

from . import dynamics
from .game import infer_game_class, require_supported_game_class

__all__ = [
    "InconclusiveStabilityWarning",
    "AnalyzedEquilibrium",
    "EquilibriumAnalysis",
    "analyze_equilibria",
    "equilibrium_table",
    "rest_points_nash",
    "rest_points_strict_nash",
    "rest_points_ess",
]


class InconclusiveStabilityWarning(RuntimeWarning):
    """Warning raised when stability cannot be classified by implemented tests."""


@dataclass
class AnalyzedEquilibrium:
    """Analysis data for one detected replicator rest point."""

    reduced_position: np.ndarray
    full_position: np.ndarray
    stability: str
    eigenvalues: Optional[np.ndarray]
    eigenvectors: Optional[np.ndarray]
    admissible_eigenvalues: Optional[np.ndarray]
    admissible_eigenvectors: Optional[np.ndarray]
    nash: Optional[bool]
    strict_nash: Optional[bool]
    ess: Optional[bool] = None
    warning: Optional[str] = None


@dataclass
class EquilibriumAnalysis:
    """Container returned by analyze_equilibria.

    Attributes
    ----------
    equilibria : list of AnalyzedEquilibrium
        One entry per detected replicator rest point in the game domain.
    game_class : str
        Supported pyNamo game-class identifier: "2Pop2S", "1Pop3S", "1Pop4S", or
        "3Pop2S".
    degenerate : bool, default=False
        True when the symbolic equilibrium solver detected a non-isolated or
        parametric equilibrium set.
    message : str or None, optional
        Warning message describing global analysis issues, usually degeneracy.

    Notes
    -----
    Table positions are full mixed strategies. Entries follow the strategy order
    supplied when defining the game. For asymmetric games, coordinates give
    action 0 (the first listed strategy) probabilities.
    """

    equilibria: List[AnalyzedEquilibrium]
    game_class: str
    degenerate: bool = False
    message: Optional[str] = None

    def to_rows(self, ndigits: int = 6) -> List[dict]:
        """Return equilibrium analysis as table rows.

        Parameters
        ----------
        ndigits : int, default=6
            Number of decimal places used when formatting numeric values.

        Returns
        -------
        rows : list of dict
            List of dictionaries with columns matching equilibrium_table.

        Notes
        -----
        The Position column follows the strategy order stored in the Game object.
        In 2Pop2S and 3Pop2S, coordinates give action 0 probabilities.
        """
        return [_equilibrium_to_row(equilibrium, ndigits) for equilibrium in self.equilibria]

    def to_dataframe(self, ndigits: int = 6):
        """Return equilibrium analysis as a pandas DataFrame.

        Parameters
        ----------
        ndigits : int, default=6
            Number of decimal places used when formatting numeric values.

        Returns
        -------
        pandas.DataFrame
            DataFrame with the same columns as equilibrium_table.
        """
        import pandas as pd

        return pd.DataFrame(self.to_rows(ndigits=ndigits))


def analyze_equilibria(game) -> EquilibriumAnalysis:
    """Annotate detected replicator rest points with static and dynamic properties.

    Parameters
    ----------
    game : pn.Game or payoff data
        Game to analyze. Passing a Game object is recommended because it
        preserves labels and metadata. Raw payoff data are also accepted.

    Returns
    -------
    EquilibriumAnalysis
        Structured equilibrium analysis including positions, stability status,
        admissible eigenvalues/eigenvectors, Nash status, strict Nash status,
        ESS status where applicable, and warnings.

    Raises
    ------
    ValueError
        If the game does not belong to a supported game class.

    Warns
    -----
    dynamics.DegenerateEquilibriumWarning
        Raised when the symbolic solver detects a non-isolated or parametric
        equilibrium set.
    InconclusiveStabilityWarning
        Raised when the implemented stability criteria do not determine the
        stability of an isolated equilibrium.

    Notes
    -----
    Returned positions follow the strategy order supplied when defining the game.
    Asymmetric coordinates give action 0 (the first listed strategy)
    probabilities.
    Eigenvalue computations are numerical. Tiny imaginary parts below the
    internal tolerance are treated as numerical noise when filtering admissible
    eigendirections at boundary equilibria.
    """
    payoff_data = _payoff_data(game)
    game_class = require_supported_game_class(game)
    if game_class == "1Pop2S":
        points, continuum = dynamics._analyze_1d(payoff_data)
        u, v = dynamics._coefficients_1d(payoff_data)
        equilibria = []
        for point in points:
            x = point.x
            full = np.array([x, 1-x])
            nash, strict = _symmetric_nash_status(full, payoff_data, 1e-12)
            eigenvalue = (1-2*x)*(u+v*x) + x*(1-x)*v
            values, vectors = np.array([eigenvalue]), np.ones((1, 1))
            equilibria.append(AnalyzedEquilibrium(
                reduced_position=np.array([x]), full_position=full,
                stability="sink" if point.stable else "source",
                eigenvalues=values, eigenvectors=vectors,
                admissible_eigenvalues=values, admissible_eigenvectors=vectors,
                nash=nash, strict_nash=strict, ess=bool(point.stable),
            ))
        return EquilibriumAnalysis(
            equilibria, game_class, degenerate=continuum,
            message="Every frequency in [0, 1] is an equilibrium; no isolated points are enumerated." if continuum else None,
        )
    raw_equilibria = dynamics.compute_equilibria(payoff_data)
    equilibria = []

    for eq in raw_equilibria:
        reduced = np.asarray(eq, dtype=float)
        if not _is_state_in_domain(reduced, game_class):
            continue

        nash, strict_nash = _nash_status(reduced, payoff_data, game_class)
        ess = _is_ess_symmetric(reduced, payoff_data, game_class)
        eig_data = _eigen_data(reduced, payoff_data, game_class)
        if eig_data is None:
            stability = "undetermined"
            eigvals = eigvecs = admissible_vals = admissible_vecs = None
            warning = "No eigenvalue analysis is implemented for this game class."
        else:
            eigvals, eigvecs = eig_data
            admissible_vals, admissible_vecs, skipped_complex = _admissible_eigenpairs(
                reduced, eig_data, game_class
            )
            stability = _classify_from_admissible_eigenvalues(
                admissible_vals, skipped_complex
            )
            warning = None

            if stability == "undetermined":
                if ess:
                    stability = "sink"
                else:
                    warning = _unresolved_warning(reduced, game_class)
                    warnings.warn(
                        warning, InconclusiveStabilityWarning, stacklevel=2
                    )

        equilibria.append(
            AnalyzedEquilibrium(
                reduced_position=reduced,
                full_position=_full_position(reduced, game_class),
                stability=stability,
                eigenvalues=eigvals,
                eigenvectors=eigvecs,
                admissible_eigenvalues=admissible_vals,
                admissible_eigenvectors=admissible_vecs,
                nash=nash,
                strict_nash=strict_nash,
                ess=ess,
                warning=warning,
            )
        )

    return EquilibriumAnalysis(
        equilibria=equilibria,
        game_class=game_class,
        degenerate=getattr(raw_equilibria, "degenerate", False),
        message=getattr(raw_equilibria, "message", None),

    )


def equilibrium_table(game, ndigits: int = 6):
    """Return a pandas table summarizing equilibrium analysis.

    Parameters
    ----------
    game : pn.Game or payoff data
        Game to analyze. Passing a Game object is recommended.
    ndigits : int, default=6
        Number of decimal places used when formatting numeric values.

    Returns
    -------
    pandas.DataFrame
        Table with columns Position, Stability Status, Nash, ESS, Strict Nash,
        Eigenvalues, Eigenvectors, and Warning.

    Notes
    -----
    The Position column follows the strategy order supplied when defining the
    game. Asymmetric coordinates give action 0 (the first listed strategy)
    probabilities.

    Only rest points returned by dynamics.compute_equilibria are classified.
    No additional Nash points or equilibrium families are introduced.
    Degeneracy metadata concerns rest-point discovery, not classification.
    """
    return analyze_equilibria(game).to_dataframe(ndigits=ndigits)


def rest_points_nash(game):
    """Return detected replicator rest points that satisfy the Nash condition.

    This filters the analyzed rest points; it does not enumerate the Nash set.

    Parameters
    ----------
    game : pn.Game or payoff data
        Game to analyze.

    Returns
    -------
    list of numpy.ndarray
        Full equilibrium positions using the same convention as
        equilibrium_table.
    """
    return [
        equilibrium.full_position
        for equilibrium in analyze_equilibria(game).equilibria
        if equilibrium.nash is True
    ]


def rest_points_strict_nash(game):
    """Return detected replicator rest points that are strict Nash.

    Parameters
    ----------
    game : pn.Game or payoff data
        Game to analyze.

    Returns
    -------
    list of numpy.ndarray
        Full equilibrium positions using the same convention as
        equilibrium_table.
    """
    return [
        equilibrium.full_position
        for equilibrium in analyze_equilibria(game).equilibria
        if equilibrium.strict_nash is True
    ]


def rest_points_ess(game):
    """Return detected replicator rest points that are ESS in symmetric games.

    Parameters
    ----------
    game : pn.Game or payoff data
        Game to analyze.

    Returns
    -------
    list of numpy.ndarray
        Full equilibrium positions using the same convention as
        equilibrium_table.

    Notes
    -----
    ESS is evaluated only for symmetric games. For asymmetric games, no ESS
    entries are returned.
    """
    return [
        equilibrium.full_position
        for equilibrium in analyze_equilibria(game).equilibria
        if equilibrium.ess is True
    ]


def _payoff_data(game):
    return getattr(game, "payoff_data", game)


def _eigen_data(reduced, payoff_data, game_class):
    if game_class == "1Pop3S":
        field = Matrix(dynamics.replicator_2p3s([x, y], 0, payoff_data))
        jacobian = field.jacobian(Matrix([x, y]))
        matrix = np.array(jacobian.subs([(x, reduced[0]), (y, reduced[1])]), dtype=float)
        return np.linalg.eig(matrix)

    if game_class == "2Pop2S":
        field = Matrix(dynamics.replicator_2p2s([x, y], 0, payoff_data))
        jacobian = field.jacobian(Matrix([x, y]))
        matrix = np.array(jacobian.subs([(x, reduced[0]), (y, reduced[1])]), dtype=float)
        return np.linalg.eig(matrix)

    if game_class == "1Pop4S":
        field = Matrix(dynamics.replicator_2p4s([x, y, z], 0, payoff_data))
        jacobian = field.jacobian(Matrix([x, y, z]))
        matrix = np.array(
            jacobian.subs([(x, reduced[0]), (y, reduced[1]), (z, reduced[2])]),
            dtype=float,
        )
        return np.linalg.eig(matrix)

    if game_class == "3Pop2S":
        matrix = _numeric_jacobian(
            lambda state: dynamics.replicator_3p2s(state, 0, payoff_data), reduced
        )
        return np.linalg.eig(matrix)

    return None


def _numeric_jacobian(field_function, point, eps: float = 1e-6):
    point = np.asarray(point, dtype=float)
    f0 = np.asarray(field_function(point), dtype=float)
    jacobian = np.zeros((point.size, point.size), dtype=float)
    for axis in range(point.size):
        if point[axis] <= eps:
            perturbed = point.copy()
            perturbed[axis] += eps
            f1 = np.asarray(field_function(perturbed), dtype=float)
            jacobian[:, axis] = (f1 - f0) / eps
        elif point[axis] >= 1.0 - eps:
            perturbed = point.copy()
            perturbed[axis] -= eps
            f1 = np.asarray(field_function(perturbed), dtype=float)
            jacobian[:, axis] = (f0 - f1) / eps
        else:
            forward = point.copy()
            backward = point.copy()
            forward[axis] += eps
            backward[axis] -= eps
            f_forward = np.asarray(field_function(forward), dtype=float)
            f_backward = np.asarray(field_function(backward), dtype=float)
            jacobian[:, axis] = (f_forward - f_backward) / (2.0 * eps)
    return jacobian


def _full_position(reduced, game_class):
    reduced = np.asarray(reduced, dtype=float)
    if game_class == "1Pop3S":
        return np.array([reduced[0], reduced[1], 1.0 - reduced.sum()])
    if game_class == "1Pop4S":
        return np.array([reduced[0], reduced[1], reduced[2], 1.0 - reduced.sum()])
    return reduced.copy()


def _nash_status(reduced, payoff_data, game_class, tol: float = 1e-8):
    """Return whether a state in the game domain is Nash and strict Nash."""
    if game_class in ("1Pop3S", "1Pop4S"):
        return _symmetric_nash_status(_full_position(reduced, game_class), payoff_data, tol)
    if game_class == "2Pop2S":
        return _asymmetric_2p2s_nash_status(reduced, payoff_data, tol)
    if game_class == "3Pop2S":
        return _three_player_two_strategy_nash_status(reduced, payoff_data, tol)
    return None, None


def _symmetric_nash_status(strategy, payoff_matrix, tol: float):
    p = np.asarray(strategy, dtype=float)
    payoff_data = payoff_matrix @ p
    return _mixed_strategy_best_response_status([p], [payoff_data], tol)


def _asymmetric_2p2s_nash_status(reduced, payoffs, tol: float):
    x_prob_action_0, y_prob_action_0 = np.asarray(reduced, dtype=float)
    p0 = np.array([x_prob_action_0, 1.0 - x_prob_action_0])
    p1 = np.array([y_prob_action_0, 1.0 - y_prob_action_0])

    payoff_player_0 = payoffs[0] @ p1
    payoff_player_1 = payoffs[1] @ p0
    return _mixed_strategy_best_response_status(
        [p0, p1], [payoff_player_0, payoff_player_1], tol
    )


def _three_player_two_strategy_nash_status(reduced, payoff_tensors, tol: float):
    probabilities_action_0 = np.asarray(reduced, dtype=float)
    mixed_strategies = [
        np.array([prob, 1.0 - prob]) for prob in probabilities_action_0
    ]

    payoff_vectors = []
    for player_index, tensor in enumerate(payoff_tensors):
        payoff_vectors.append(
            np.array(
                [
                    _expected_payoff_for_action(
                        tensor, probabilities_action_0, player_index, 0
                    ),
                    _expected_payoff_for_action(
                        tensor, probabilities_action_0, player_index, 1
                    ),
                ]
            )
        )

    return _mixed_strategy_best_response_status(mixed_strategies, payoff_vectors, tol)


def _expected_payoff_for_action(payoff_tensor, probabilities_action_0, player_index, action):
    total = 0.0
    for a0 in (0, 1):
        for a1 in (0, 1):
            for a2 in (0, 1):
                actions = [a0, a1, a2]
                if actions[player_index] != action:
                    continue

                probability = 1.0
                for idx, action_idx in enumerate(actions):
                    if idx == player_index:
                        continue
                    prob_action_0 = probabilities_action_0[idx]
                    probability *= prob_action_0 if action_idx == 0 else 1.0 - prob_action_0

                total += payoff_tensor[a0, a1, a2] * probability
    return total


def _mixed_strategy_best_response_status(strategies, payoff_vectors, tol: float):
    is_nash = True
    is_strict = True

    for strategy, payoff_data in zip(strategies, payoff_vectors):
        strategy = np.asarray(strategy, dtype=float)
        payoff_data = np.asarray(payoff_data, dtype=float)

        support = strategy > tol
        max_payoff = np.max(payoff_data)
        if np.any(np.abs(payoff_data[support] - max_payoff) > tol):
            is_nash = False

        if np.count_nonzero(support) != 1:
            is_strict = False
        else:
            chosen = int(np.flatnonzero(support)[0])
            alternatives = np.delete(payoff_data, chosen)
            if alternatives.size and not np.all(payoff_data[chosen] > alternatives + tol):
                is_strict = False

    return bool(is_nash), bool(is_nash and is_strict)


def _is_state_in_domain(reduced, game_class, tol: float = 1e-9) -> bool:
    reduced = np.asarray(reduced, dtype=float)
    if game_class == "1Pop2S":
        return reduced.shape == (1,) and np.all(reduced >= -tol) and np.all(reduced <= 1.0 + tol)
    if game_class in ("1Pop3S", "1Pop4S"):
        return np.all(reduced >= -tol) and reduced.sum() <= 1.0 + tol
    if game_class in ("2Pop2S", "3Pop2S"):
        return np.all(reduced >= -tol) and np.all(reduced <= 1.0 + tol)
    return False


def _is_interior_point(point, game_class, tol: float = 1e-9) -> bool:
    point = np.asarray(point, dtype=float)
    if game_class in ("1Pop3S", "1Pop4S"):
        return np.all(point > tol) and point.sum() < 1.0 - tol
    if game_class in ("2Pop2S", "3Pop2S"):
        return np.all(point > tol) and np.all(point < 1.0 - tol)
    return False


def _is_admissible_direction(point, direction, game_class, tol: float = 1e-9) -> bool:
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)

    if game_class == "1Pop3S":
        if abs(point[0]) <= tol and direction[0] < -tol:
            return False
        if abs(point[1]) <= tol and direction[1] < -tol:
            return False
        if abs(1.0 - point.sum()) <= tol and direction.sum() > tol:
            return False
        return True

    if game_class in ("2Pop2S", "3Pop2S"):
        for coord, delta in zip(point, direction):
            if abs(coord) <= tol and delta < -tol:
                return False
            if abs(1.0 - coord) <= tol and delta > tol:
                return False
        return True

    if game_class == "1Pop4S":
        if any(abs(point[idx]) <= tol and direction[idx] < -tol for idx in range(3)):
            return False
        if abs(1.0 - point.sum()) <= tol and direction.sum() > tol:
            return False
        return True

    return False


def _admissible_eigenpairs(point, eig_data, game_class, tol: float = 1e-7):
    eigvals, eigvecs = eig_data
    if _is_interior_point(point, game_class, tol):
        return np.asarray(eigvals, dtype=complex), np.asarray(eigvecs), False

    values = []
    vectors = []
    skipped_complex_boundary = False
    for idx, eigval in enumerate(eigvals):
        eigvec = eigvecs[:, idx]
        if abs(np.imag(eigval)) <= tol:
            eigval = complex(np.real(eigval), 0.0)

        if np.linalg.norm(np.imag(eigvec)) > tol:
            skipped_complex_boundary = True
            continue

        direction = np.real(eigvec)
        if (
            _is_admissible_direction(point, direction, game_class, tol)
            or _is_admissible_direction(point, -direction, game_class, tol)
        ):
            values.append(eigval)
            vectors.append(direction)

    values, vectors = _complete_repeated_boundary_eigenspaces(
        point,
        eigvals,
        eigvecs,
        values,
        vectors,
        game_class,
        tol,
    )

    if vectors:
        vectors = np.column_stack(vectors)
    else:
        vectors = np.empty((len(point), 0))
    return np.asarray(values, dtype=complex), vectors, skipped_complex_boundary


def _complete_repeated_boundary_eigenspaces(
    point,
    eigvals,
    eigvecs,
    values,
    vectors,
    game_class,
    tol,
):
    """Add admissible directions missed by arbitrary repeated-eigenvalue bases."""
    candidates = _admissible_direction_candidates(point, game_class)
    if not candidates:
        return values, vectors

    used = [False] * len(eigvals)
    for idx, eigval in enumerate(eigvals):
        if used[idx]:
            continue

        group = [
            j
            for j, other in enumerate(eigvals)
            if abs(eigval - other) <= tol
        ]
        for j in group:
            used[j] = True

        if len(group) <= 1:
            continue

        kept_count = sum(abs(eigval - val) <= tol for val in values)
        missing_count = len(group) - kept_count
        if missing_count <= 0:
            continue

        eigenspace = eigvecs[:, group]
        for candidate in candidates:
            if missing_count <= 0:
                break
            if not _direction_in_span(candidate, eigenspace, tol):
                continue
            if _is_redundant_direction(candidate, vectors, tol):
                continue
            values.append(complex(np.real(eigval), 0.0) if abs(np.imag(eigval)) <= tol else eigval)
            vectors.append(candidate)
            missing_count -= 1

    return values, vectors


def _admissible_direction_candidates(point, game_class):
    point = np.asarray(point, dtype=float)

    if game_class == "1Pop3S":
        full = _full_position(point, game_class)
        candidates = []
        for absent_strategy in np.flatnonzero(np.isclose(full, 0.0)):
            for present_strategy in np.flatnonzero(full > 0.0):
                direction_full = np.zeros(3)
                direction_full[absent_strategy] = 1.0
                direction_full[present_strategy] = -1.0
                candidates.append(direction_full[:2])
        return candidates

    if game_class == "1Pop4S":
        full = _full_position(point, game_class)
        candidates = []
        for absent_strategy in np.flatnonzero(np.isclose(full, 0.0)):
            for present_strategy in np.flatnonzero(full > 0.0):
                direction_full = np.zeros(4)
                direction_full[absent_strategy] = 1.0
                direction_full[present_strategy] = -1.0
                candidates.append(direction_full[:3])
        return candidates

    return []


def _direction_in_span(direction, basis, tol):
    direction = np.asarray(direction, dtype=float)
    basis = _real_direction_basis(basis, tol)
    if basis.size == 0:
        return False
    coeffs, *_ = np.linalg.lstsq(basis, direction, rcond=None)
    projection = basis @ coeffs
    return np.linalg.norm(projection - direction) <= tol


def _real_direction_basis(basis, tol):
    """Convert real or complex eigenvectors into real admissible directions."""
    basis = np.asarray(basis)
    if basis.size == 0:
        return np.empty((basis.shape[0], 0), dtype=float)

    columns = []
    for idx in range(basis.shape[1]):
        vector = basis[:, idx]
        real_part = np.real(vector)
        imag_part = np.imag(vector)

        if np.linalg.norm(real_part) > tol:
            columns.append(real_part)
        if np.linalg.norm(imag_part) > tol:
            columns.append(imag_part)

    if not columns:
        return np.empty((basis.shape[0], 0), dtype=float)
    return np.column_stack(columns).astype(float)


def _is_redundant_direction(direction, vectors, tol):
    if not vectors:
        return False
    existing = np.column_stack(vectors)
    return _direction_in_span(direction, existing, tol)


def _classify_from_admissible_eigenvalues(admissible, skipped_complex_boundary, tol: float = 1e-9):
    if admissible is None or admissible.size == 0:
        return "undetermined"

    real_parts = np.real(admissible)
    if np.all(real_parts < -tol):
        return "sink"
    if np.all(real_parts > tol):
        return "source"
    if np.any(real_parts > tol) and np.any(real_parts < -tol):
        return "saddle"
    if np.all(np.abs(real_parts) <= tol):
        if skipped_complex_boundary:
            return "undetermined"
        if np.any(np.abs(np.imag(admissible)) > tol):
            return "center"
        return "undetermined"

    if np.any(real_parts > tol):
        # Positive real parts prove instability, but zero real parts prevent a
        # sharper source/saddle classification by linearization alone.
        return "unstable"

    # Negative real parts together with zero real parts do not prove asymptotic
    # stability. A later ESS check may still classify symmetric-game cases.
    return "undetermined"


def _is_ess_symmetric(reduced, payoff_data, game_class, tol: float = 1e-8) -> Optional[bool]:
    if game_class not in ("1Pop3S", "1Pop4S"):
        return None

    p = np.clip(_full_position(reduced, game_class), 0.0, 1.0)
    total = p.sum()
    if total <= tol:
        return False
    p = p / total

    pure_payoff_data = payoff_data @ p
    payoff_at_p = float(p @ pure_payoff_data)
    if np.any(pure_payoff_data > payoff_at_p + tol):
        return False

    support = np.flatnonzero(p > tol)
    best_responses = np.flatnonzero(np.abs(pure_payoff_data - payoff_at_p) <= tol)
    return _ess_on_critical_cone(payoff_data, support, best_responses, tol)


def _tangent_basis(size):
    if size <= 1:
        return np.empty((size, 0))
    basis = np.zeros((size, size - 1))
    for idx in range(size - 1):
        basis[idx, idx] = 1.0
        basis[-1, idx] = -1.0
    return basis


def _unresolved_warning(reduced, game_class):
    full = _full_position(reduced, game_class)
    return (
        "Stability classification is inconclusive for equilibrium "
        f"{np.round(full, 10).tolist()}: admissible linearization has "
        "eigenvalues with zero real part or unsupported eigendirections, "
        "and the implemented fallback did not establish asymptotic stability."
    )


def _equilibrium_to_row(equilibrium: AnalyzedEquilibrium, ndigits: int) -> dict:
    return {
        "Position": _format_array(equilibrium.full_position, ndigits),
        "Stability Status": equilibrium.stability,
        "Nash": None if equilibrium.nash is None else bool(equilibrium.nash),
        "ESS": "NA" if equilibrium.ess is None else bool(equilibrium.ess),
        "Strict Nash": None if equilibrium.strict_nash is None else bool(equilibrium.strict_nash),
        "Eigenvalues": _format_array(equilibrium.admissible_eigenvalues, ndigits),
        "Eigenvectors": _format_matrix(equilibrium.admissible_eigenvectors, ndigits),
        "Warning": equilibrium.warning,
    }


def _format_number(value, ndigits: int):
    if value is None:
        return None
    value = complex(value)
    real = round(float(value.real), ndigits)
    imag = round(float(value.imag), ndigits)
    if imag == 0:
        return real
    return complex(real, imag)


def _format_array(values, ndigits: int):
    if values is None:
        return None
    return [_format_number(value, ndigits) for value in np.asarray(values).ravel()]


def _format_matrix(values, ndigits: int):
    if values is None:
        return None
    matrix = np.asarray(values)
    return [
        [_format_number(matrix[row, col], ndigits) for row in range(matrix.shape[0])]
        for col in range(matrix.shape[1])
    ]


def _ess_on_critical_cone(a, support, best, tol=1e-8):
    """Test z'Az<0 on sum(z)=0, z_unused>=0, z_nonbest=0.

    The sphere maximum lies on a cone face and is an eigenvector of the
    symmetric quadratic form restricted to that face's linear span. Enumerate
    faces and check nonnegative eigenvalue eigenspaces for feasible directions.
    """
    from itertools import combinations
    from scipy.linalg import null_space
    from scipy.optimize import linprog
    unused = [i for i in best if i not in support]
    for count in range(len(unused) + 1):
        for entering in combinations(unused, count):
            indices = list(support) + list(entering)
            if len(indices) < 2:
                continue
            basis = null_space(np.ones((1, len(indices))))
            submatrix = a[np.ix_(indices, indices)]
            restricted = basis.T @ ((submatrix + submatrix.T) / 2) @ basis
            eigenvalues, eigenvectors = np.linalg.eigh(restricted)
            for value in eigenvalues:
                if value < -tol:
                    continue
                group = np.abs(eigenvalues - value) <= tol
                space = basis @ eigenvectors[:, group]
                # Homogeneity lets us scale any nonzero feasible direction
                # until one coordinate equals +1 or -1. Check both signs.
                constraints = -space[len(support):]
                for row in space:
                    for sign in (-1.0, 1.0):
                        result = linprog(
                            np.zeros(space.shape[1]),
                            A_ub=constraints if len(constraints) else None,
                            b_ub=np.zeros(len(constraints)) if len(constraints) else None,
                            A_eq=row[None, :],
                            b_eq=[sign],
                            bounds=[(None, None)] * space.shape[1],
                            method="highs",
                        )
                        if result.success:
                            return False
                        if result.status != 2:
                            raise RuntimeError(
                                "ESS cone feasibility test failed: " + result.message
                            )
    return True
