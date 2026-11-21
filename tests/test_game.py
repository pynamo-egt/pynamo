import numpy as np
import pytest

from pynamo_egt import analyze_equilibria, dynamics, game
from pynamo_egt.examples import games as catalogue


def test_symmetric_3_strategy_game_class():
    g = game.Game(
        "RPS",
        np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float),
        strategy_labels=["R", "P", "S"],
    )

    assert g.symmetric is True
    assert g.game_class == "1Pop3S"
    assert g.num_strategies() == 3
    assert g.strategy_labels == ["R", "P", "S"]


def test_asymmetric_2_strategy_game_class():
    g = game.Game(
        "Matching Pennies",
        (
            np.array([[1, -1], [-1, 1]], dtype=float),
            np.array([[-1, 1], [1, -1]], dtype=float),
        ),
        symmetric=False,
    )

    assert g.symmetric is False
    assert g.game_class == "2Pop2S"
    assert g.num_strategies() == 2


def test_incompatible_symmetry_flag_raises():
    with pytest.raises(ValueError):
        game.Game(
            "Bad symmetry flag",
            (
                np.array([[1, 0], [0, 1]], dtype=float),
                np.array([[1, 0], [0, 1]], dtype=float),
            ),
            symmetric=True,
        )


def test_expected_payoffs_for_symmetric_game():
    g = game.Game(
        "Coordination",
        np.array([[1, 0], [0, 1]], dtype=float),
    )

    np.testing.assert_allclose(g.expected_payoffs([[0.25, 0.75]]), [0.25, 0.75])


@pytest.mark.parametrize('size,expected', [(2, '1Pop2S'), (3, '1Pop3S'), (4, '1Pop4S')])
def test_population_class_for_symmetric_games(size, expected):
    assert game.Game('symmetric', np.eye(size)).game_class == expected


def test_three_population_class():
    assert game.Game('three populations', tuple(np.zeros((2, 2, 2)) for _ in range(3))).game_class == '3Pop2S'


@pytest.mark.parametrize('old,new', [
    ('2P2S', '2Pop2S'), ('2P3S', '1Pop3S'),
    ('2P4S', '1Pop4S'), ('3P2S', '3Pop2S'),
])
def test_legacy_catalogue_lookup(old, new):
    from pynamo_egt.examples import games
    current = games.by_class(new)
    assert current
    assert games.by_class(old) == current
    assert all(g.game_class == new for g in current.values())


@pytest.mark.parametrize(
    "payoffs, message",
    [
        ((np.eye(2), np.eye(2), np.eye(2)), r"3 payoff arrays of rank 2.*3 players.*rank 3"),
        ((np.zeros((2, 2, 2)), np.zeros((2, 2, 2))), r"2 payoff arrays of rank 3.*2 players.*rank 2"),
        ((np.eye(2),), r"1 payoff arrays? of rank 2"),
    ],
    ids=["three-matrices", "two-tensors", "one-element-tuple"],
)
def test_payoff_rank_must_match_number_of_players(payoffs, message):
    with pytest.raises(ValueError, match=message):
        game.Game("Malformed", payoffs)


def test_payoff_arrays_must_share_one_shape():
    with pytest.raises(ValueError, match=r"same shape.*\(2, 2\).*\(3, 3\)"):
        game.Game("Unequal", (np.eye(2), np.eye(3)))


def test_single_payoff_array_must_be_a_matrix():
    with pytest.raises(ValueError, match=r"square matrix.*\(3, 2, 2, 2\).*tuple"):
        game.Game("Stacked tensors", np.zeros((3, 2, 2, 2)))


def test_every_catalogue_game_has_a_supported_class():
    from pynamo_egt import examples

    for _, g in examples.games.items():
        assert g.game_class != "unsupported"


def test_game_names_is_deprecated_but_still_works():
    rps = game.Game("RPS", np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float))
    with pytest.warns(DeprecationWarning, match="game_names is deprecated"):
        assert game.game_names([(7, rps)]) == {7: "RPS"}


CATALOGUE = list(catalogue.items())
TWO_POPULATION = list(catalogue.by_class("2Pop2S").items())


@pytest.mark.parametrize("name, legacy", TWO_POPULATION, ids=[n for n, _ in TWO_POPULATION])
def test_bimatrix_matches_legacy_focal_player_game(name, legacy):
    A, focal_B = legacy.payoff_data
    built = game.Game.bimatrix(A, focal_B.T, name=legacy.name)

    for ours, theirs in zip(built.payoff_data, legacy.payoff_data):
        np.testing.assert_array_equal(ours, theirs)
    for state in np.random.default_rng(0).random((20, 2)):
        np.testing.assert_allclose(
            dynamics.replicator_2p2s(state, 0, built.payoff_data),
            dynamics.replicator_2p2s(state, 0, legacy.payoff_data),
        )
    assert analyze_equilibria(built).to_rows() == analyze_equilibria(legacy).to_rows()


def test_bimatrix_uses_textbook_layout():
    # Player 2 earns B[i, j] when player 1 plays i and player 2 plays j.
    A = np.array([[3.0, 0.0], [5.0, 1.0]])
    B = np.array([[3.0, 5.0], [0.0, 1.0]])
    g = game.Game.bimatrix(A, B, [["C", "D"], ["C", "D"]], name="PD", player_labels=["Row", "Column"])

    u1, u2 = g.payoffs([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(u1, A[:, 1])
    np.testing.assert_allclose(u2, B[0, :])
    assert g.name == "PD" and g.symmetric is False
    assert g.populations == (2, 2)
    assert g.labels == (("C", "D"), ("C", "D"))
    assert g.strategy_labels == ["C", "D"]
    assert g.player_labels == ["Row", "Column"]
    np.testing.assert_array_equal(g.matrices[1], B.T)


def test_symmetric_matches_legacy_game():
    matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    built = game.Game.symmetric(matrix, ["R", "P", "S"], name="RPS", description="cyclic")
    legacy = game.Game("RPS", matrix, strategy_labels=["R", "P", "S"])

    np.testing.assert_array_equal(built.payoff_data, legacy.payoff_data)
    assert built.symmetric is True and built.game_class == "1Pop3S"
    assert built.labels == legacy.labels == (("R", "P", "S"),)
    assert built.description == "cyclic"


@pytest.mark.parametrize("name, g", CATALOGUE, ids=[n for n, _ in CATALOGUE])
def test_legacy_games_expose_populations_and_matrices(name, g):
    assert g.is_linear is True
    assert len(g.populations) == len(g.labels) == len(g.matrices)
    assert all(len(labels) == n for labels, n in zip(g.labels, g.populations))


@pytest.mark.parametrize("name, g", CATALOGUE, ids=[n for n, _ in CATALOGUE])
def test_batched_payoffs_match_expected_payoffs(name, g):
    rng = np.random.default_rng(1)
    states = [rng.dirichlet(np.ones(n), size=(4, 3)) for n in g.populations]
    batched = g.payoffs(states)

    assert [u.shape for u in batched] == [(4, 3, n) for n in g.populations]
    for index in np.ndindex(4, 3):
        expected = g.expected_payoffs([s[index] for s in states])
        np.testing.assert_allclose(np.vstack([u[index] for u in batched]), np.atleast_2d(expected))


@pytest.mark.parametrize("game_class, field", [
    ("1Pop3S", "replicator_2p3s"), ("1Pop4S", "replicator_2p4s"),
    ("2Pop2S", "replicator_2p2s"), ("3Pop2S", "replicator_3p2s"),
])
def test_payoffs_drive_the_existing_replicator_field(game_class, field):
    rng = np.random.default_rng(2)
    for g in catalogue.by_class(game_class).values():
        states = [rng.dirichlet(np.ones(n)) for n in g.populations]
        velocity = [x * (u - x @ u) for x, u in zip(states, g.payoffs(states))]
        reduced = [x[:-1] for x in states] if g.symmetric else [[x[0]] for x in states]
        expected = np.concatenate([v[:-1] if g.symmetric else v[:1] for v in velocity])
        actual = getattr(dynamics, field)(np.concatenate(reduced), 0, g.payoff_data)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("build, message", [
    (lambda: game.Game.symmetric([[1, 0, 0], [0, 1, 0]]), r"square 2-D matrix.*\(2, 3\)"),
    (lambda: game.Game.symmetric([1, 2]), r"square 2-D matrix.*\(2,\)"),
    (lambda: game.Game.symmetric(np.eye(2), ["A"]), r"labels must name each of the 2 strategies; got 1"),
    (lambda: game.Game.bimatrix(np.eye(2), np.eye(3)), r"same shape.*\(2, 2\) and \(3, 3\)"),
    (lambda: game.Game.bimatrix(np.eye(2), [1, 2]), r"2-D matrices.*\(2,\)"),
    (lambda: game.Game.bimatrix(np.ones((2, 3)), np.ones((2, 3))), r"same number of strategies.*\(2, 3\)"),
    (lambda: game.Game.bimatrix(np.eye(2), np.eye(2), ["H", "D"]), r"pair of strategy-label lists"),
    (lambda: game.Game.bimatrix(np.eye(2), np.eye(2), [["H", "D"]]), r"pair of strategy-label lists"),
    (lambda: game.Game.bimatrix(np.eye(2), np.eye(2), [["H", "D"], ["H"]]), r"player 2 must name each of the 2"),
], ids=["symmetric-rectangular", "symmetric-1d", "symmetric-labels", "bimatrix-shapes",
        "bimatrix-1d", "bimatrix-rectangular", "bimatrix-flat-labels", "bimatrix-one-list",
        "bimatrix-label-count"])
def test_constructors_reject_invalid_input(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_payoffs_rejects_mismatched_states():
    g = game.Game.bimatrix(np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match=r"Expected 2 population"):
        g.payoffs([[0.5, 0.5]])
    with pytest.raises(ValueError, match=r"population 2 must have shape \(\.\.\., 2\)"):
        g.payoffs([[0.5, 0.5], [1.0, 0.0, 0.0]])
