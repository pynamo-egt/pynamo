import numpy as np
import pytest

from pynamo_egt import dynamics, examples

from sympy_oracle import sympy_jacobian

SHAPES = {"1Pop3S": (3, 3), "1Pop4S": (4, 4), "2Pop2S": (2, 2, 2), "3Pop2S": (3, 2, 2, 2)}


def _random_game(game_class: str, rng: np.random.Generator):
    payoff = np.round(rng.normal(size=SHAPES[game_class]), 2)
    return payoff if game_class.startswith("1Pop") else tuple(payoff)


@pytest.mark.parametrize("game_class", sorted(SHAPES))
def test_closed_form_jacobian_matches_symbolic_derivative(game_class):
    rng = np.random.default_rng(sorted(SHAPES).index(game_class))
    dimension = {"1Pop3S": 2, "1Pop4S": 3, "2Pop2S": 2, "3Pop2S": 3}[game_class]
    for _ in range(10):
        payoff = _random_game(game_class, rng)
        # Points outside the state space too: the Jacobian is a polynomial identity.
        point = rng.uniform(-0.5, 1.5, size=dimension)
        np.testing.assert_allclose(
            dynamics.replicator_jacobian(point, payoff),
            sympy_jacobian(payoff, point),
            rtol=1e-10,
            atol=1e-12,
        )


def test_closed_form_jacobian_matches_symbolic_derivative_on_catalogue_rest_points():
    for name, game in examples.games.items():
        if game.game_class == "1Pop2S":
            continue
        for point in dynamics.compute_equilibria(game.payoff_data):
            np.testing.assert_allclose(
                dynamics.replicator_jacobian(np.array(point), game.payoff_data),
                sympy_jacobian(game.payoff_data, point),
                rtol=1e-10,
                atol=1e-12,
                err_msg=name,
            )


def test_jacobian_rejects_one_dimensional_games():
    with pytest.raises(ValueError, match="1Pop2S"):
        dynamics.replicator_jacobian(np.array([0.5]), np.eye(2))
