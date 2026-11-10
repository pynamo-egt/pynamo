import time
import warnings
from typing import Iterator

import numpy as np
import pytest

from pynamo_egt import dynamics, examples
from pynamo_egt._rest_points import to_fraction

from sympy_oracle import sympy_rest_points


def _random_games(n: int, count: int, seed: int, integers: bool = True) -> Iterator[np.ndarray]:
    """Yield seeded random n×n payoff matrices (small integers or 2-decimal floats)."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        if integers:
            yield rng.integers(-3, 4, size=(n, n)).astype(float)
        else:
            yield np.round(rng.normal(size=(n, n)), 2)


def _symmetric_velocity(payoff_matrix: np.ndarray, reduced: np.ndarray) -> np.ndarray:
    full = np.append(reduced, 1 - np.sum(reduced))
    fitness = payoff_matrix @ full
    return full * (fitness - full @ fitness)


def _compute(payoff_data) -> dynamics.EquilibriumResult:
    """Rest points with the degeneracy warning silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", dynamics.DegenerateEquilibriumWarning)
        return dynamics.compute_equilibria(payoff_data)


def test_payoffs_are_read_at_15_significant_digits():
    assert to_fraction(0.1 + 0.2) == to_fraction(0.3)


def test_four_strategy_game_that_hung_sympy_finishes_quickly():
    payoff = np.array(
        [[-3, -3, -2, 3], [1, 1, 3, 0], [-3, -2, -2, 1], [-1, -1, -3, 1]], dtype=float
    )
    start = time.perf_counter()
    with pytest.warns(dynamics.DegenerateEquilibriumWarning):
        result = dynamics.compute_equilibria(payoff)
    assert time.perf_counter() - start < 1.0
    assert result.degenerate
    assert [1.0, 0.0, 0.0] in result and [0.0, 0.0, 0.0] in result


@pytest.mark.parametrize(
    "n, seed, integers",
    [(3, 1, True), (3, 2, False), (4, 3, True), (4, 4, False)],
)
def test_reported_rest_points_have_zero_velocity(n, seed, integers):
    for payoff in _random_games(n, 25, seed, integers):
        for point in _compute(payoff):
            assert np.allclose(_symmetric_velocity(payoff, np.array(point)), 0, atol=1e-9)


@pytest.mark.parametrize(
    "n, count, seed, integers",
    [(3, 15, 10, True), (3, 10, 11, False), (4, 6, 12, True), (4, 4, 13, False)],
)
def test_matches_sympy_on_games_without_degenerate_ties(n, count, seed, integers):
    compared = 0
    for payoff in _random_games(n, count, seed, integers):
        expected, isolated = sympy_rest_points(payoff)
        result = _compute(payoff)
        if not isolated or result.degenerate:
            continue
        assert list(result) == expected
        compared += 1
    assert compared >= count // 2


def test_matches_sympy_on_catalogue_games():
    for name, game in examples.games.items():
        if game.game_class not in ("1Pop3S", "1Pop4S"):
            continue
        expected, isolated = sympy_rest_points(game.payoff_data)
        if isolated:
            assert list(_compute(game.payoff_data)) == expected, name
