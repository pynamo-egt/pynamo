import numpy as np
import pytest

from pynamo_egt import analysis
from pynamo_egt import dynamics
from pynamo_egt import game
from pynamo_egt import examples
EXPECTED_COLUMNS = [
    "Position",
    "Stability Status",
    "Nash",
    "ESS",
    "Strict Nash",
    "Eigenvalues",
    "Eigenvectors",
    "Warning",
]


def test_analysis_rows_have_expected_columns():
    g = examples.games.good_rps
    rows = analysis.analyze_equilibria(g).to_rows()

    assert rows
    assert list(rows[0].keys()) == EXPECTED_COLUMNS


def test_matching_pennies_interior_equilibrium_is_found():
    g = examples.games.matching_pennies
    rows = analysis.analyze_equilibria(g).to_rows()
    positions = np.asarray([row["Position"] for row in rows], dtype=float)

    assert np.any(np.all(np.isclose(positions, [0.5, 0.5]), axis=1))


def test_degenerate_equilibrium_warning_is_emitted():
    degenerate_edge_game = game.Game(
        "Degenerate Edge",
        np.array(
            [
                [1, 1, 0],
                [1, 1, 0],
                [2, 2, 0],
            ],
            dtype=float,
        ),
        strategy_labels=["A", "B", "C"],
    )

    with pytest.warns(dynamics.DegenerateEquilibriumWarning):
        result = analysis.analyze_equilibria(degenerate_edge_game)

    assert result.degenerate is True
    assert result.message is not None


def test_static_equilibrium_concepts_are_present():
    g = examples.games.good_rps
    equilibrium = analysis.analyze_equilibria(g).equilibria[0]

    assert isinstance(equilibrium.nash, bool)
    assert isinstance(equilibrium.strict_nash, bool)
    assert isinstance(equilibrium.ess, bool)


def test_one_population_boolean_columns_use_python_booleans():
    result = analysis.analyze_equilibria(examples.games.hawk_dove)

    for equilibrium, row in zip(result.equilibria, result.to_rows()):
        assert isinstance(equilibrium.nash, bool)
        assert isinstance(equilibrium.strict_nash, bool)
        assert isinstance(equilibrium.ess, bool)
        assert isinstance(row["Nash"], bool)
        assert isinstance(row["Strict Nash"], bool)
        assert isinstance(row["ESS"], bool)


def test_hawk_dove_retaliator_edge_ess_has_eigenpairs():
    result = analysis.analyze_equilibria(examples.games.hawk_dove_retaliator)
    equilibrium = next(
        eq
        for eq in result.equilibria
        if np.allclose(eq.full_position, [2 / 3, 1 / 3, 0])
    )

    assert equilibrium.ess is True
    assert equilibrium.stability == "sink"
    assert equilibrium.admissible_eigenvalues.size > 0
    assert equilibrium.admissible_eigenvectors.shape[1] > 0


def test_3pop2s_first_strategy_dominance_has_sink_at_one():
    payoff_tensors = []
    for player in range(3):
        tensor = np.zeros((2, 2, 2), dtype=float)
        for action_profile in np.ndindex(tensor.shape):
            tensor[action_profile] = 1.0 if action_profile[player] == 0 else 0.0
        payoff_tensors.append(tensor)

    result = analysis.analyze_equilibria(
        game.Game("Dominant first strategies", tuple(payoff_tensors), symmetric=False)
    )
    equilibrium = next(
        eq for eq in result.equilibria
        if np.allclose(eq.reduced_position, [1.0, 1.0, 1.0])
    )

    assert equilibrium.nash is True
    assert equilibrium.strict_nash is True
    assert equilibrium.stability == "sink"


def test_analysis_rejects_unsupported_game_size():
    five_strategies = game.Game("Five strategies", np.eye(5))
    with pytest.raises(ValueError, match=r"shape \(5, 5\).*1Pop2S.*1Pop3S.*1Pop4S.*2Pop2S.*3Pop2S"):
        analysis.analyze_equilibria(five_strategies)
    with pytest.raises(ValueError, match="Supported game classes"):
        analysis.equilibrium_table(five_strategies)
    with pytest.raises(ValueError, match="Supported game classes"):
        dynamics.compute_equilibria(five_strategies.payoff_data)
