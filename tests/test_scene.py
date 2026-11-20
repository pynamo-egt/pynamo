import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PatchCollection, PathCollection

import pynamo_egt as pn
from pynamo_egt import dynamics, examples

TRIANGLE_GAMES = examples.games.by_class("1Pop3S")


def test_scene_is_json_serialisable_and_deterministic():
    first = pn.scene(examples.games.good_rps)
    assert json.loads(json.dumps(first)) == first
    assert pn.scene(examples.games.good_rps) == first
    assert first["version"] == 1 and first["state_space"] == {"kind": "triangle", "populations": [3]}


def test_replicator_velocity_and_jacobian_match_legacy_field():
    game = examples.games.good_rps
    rule = pn.Replicator()
    states = np.random.default_rng(0).dirichlet(np.ones(3), size=(5, 4))[..., :2]
    velocities = rule.velocity(game, states)
    assert velocities.shape == states.shape
    for state, velocity in zip(states.reshape(-1, 2), velocities.reshape(-1, 2)):
        np.testing.assert_allclose(velocity, dynamics.replicator_2p3s(state, 0, game.payoff_data))
        np.testing.assert_allclose(rule.jacobian(game, state), dynamics.replicator_jacobian(state, game.payoff_data))
    assert (rule.smooth, rule.boundary_invariant, rule.nash_stationary) == (True, True, False)


def test_state_space_round_trips_and_places_vertices_like_phase_portrait():
    space = pn.StateSpace((3,))
    reduced = np.random.default_rng(1).dirichlet(np.ones(3), size=7)[:, :2]
    np.testing.assert_allclose(space.to_reduced(space.to_full(reduced)), reduced)
    np.testing.assert_allclose(space.to_full(reduced).sum(axis=-1), 1.0)
    np.testing.assert_allclose(
        space.embed(np.eye(3)[:, :2]),
        [pn.drawer.simplex_to_plane_2p3s(*vertex[:2]) for vertex in np.eye(3)],
    )


@pytest.mark.parametrize("name", sorted(TRIANGLE_GAMES))
def test_scene_rest_points_equal_analysis(name):
    game = TRIANGLE_GAMES[name]
    expected = pn.analyze_equilibria(game).equilibria
    actual = pn.scene(game)["rest_points"]
    assert [point["stability"] for point in actual] == [point.stability for point in expected]
    np.testing.assert_allclose([point["position"] for point in actual], [point.full_position for point in expected])


def _drawn(ax):
    lines = sorted(np.round(line.get_xydata(), 10).tobytes() for line in ax.lines)
    markers = sorted(
        (np.round(c.get_offsets(), 10).tobytes(), tuple(np.round(c.get_facecolor(), 6).ravel()))
        for c in ax.collections if isinstance(c, PathCollection)
    )
    heads = sorted(
        np.round(path.vertices, 10).tobytes()
        for c in ax.collections if isinstance(c, PatchCollection) for path in c.get_paths()
    )
    labels = sorted((t.get_text(), np.round(t.xyann, 10).tobytes()) for t in ax.texts)
    return lines, markers, heads, labels


@pytest.mark.parametrize("name", sorted(TRIANGLE_GAMES))
@pytest.mark.parametrize("options", [{}, {"starts": [[0.2, 0.3], [0.6, 0.1]]}, {"random_state": 3}], ids=["default", "starts", "seeded"])
def test_renderer_draws_what_phase_portrait_draws(name, options):
    game = TRIANGLE_GAMES[name]
    _, legacy = pn.phase_portrait(game, show_speed=False, **options)
    _, rendered = pn.render.matplotlib(pn.scene(game, **options))
    try:
        assert _drawn(rendered) == _drawn(legacy)
        assert rendered.get_title() == legacy.get_title()
    finally:
        plt.close("all")


def test_renderer_warns_once_about_unstable_points():
    game = TRIANGLE_GAMES["hawk_dove_retaliator"]
    with pytest.warns(pn.drawer.PlottingWarning) as record:
        pn.render.matplotlib(pn.scene(game, trajectory_number=1))
    plotting = [w for w in record if issubclass(w.category, pn.drawer.PlottingWarning)]
    assert len(plotting) == 1
    assert plotting[0].filename == __file__
    plt.close("all")


def test_renderer_rejects_unknown_style_options():
    with pytest.raises(TypeError, match="trajectory_colour"):
        pn.render.matplotlib(pn.scene(examples.games.good_rps, trajectory_number=1), trajectory_colour="red")
    plt.close("all")


def test_scenes_for_other_classes_are_not_available_yet():
    with pytest.raises(NotImplementedError, match="2Pop2S"):
        pn.scene(examples.games.matching_pennies)
