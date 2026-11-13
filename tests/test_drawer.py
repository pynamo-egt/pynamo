import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PatchCollection

from pynamo_egt import drawer
from pynamo_egt import examples
from pynamo_egt.game import Game


def test_default_trajectory_starts_are_deterministic_grids():
    triangle = drawer._default_starts("1Pop3S", random_state=None, trajectory_number=2)
    square = drawer._default_starts("2Pop2S", random_state=None, trajectory_number=2)

    np.testing.assert_allclose(triangle, [[1 / 6, 1 / 2], [1 / 2, 1 / 6]])
    np.testing.assert_allclose(square, [[1 / 4, 1 / 2], [3 / 4, 1 / 2]])


def test_1pop3s_default_edge_flow_draws_one_midpoint_arrow_per_rps_edge():
    fig, ax = drawer.phase_portrait(
        examples.games.standard_rps,
        starts=[],
        show_edge_flow=True,
        trajectory_arrows=None,
        show_speed=False,
        show_equilibria=False,
    )

    heads = [artist for artist in ax.collections if isinstance(artist, PatchCollection)]
    assert len(heads) == 3
    plt.close(fig)


def test_edge_flow_ignores_time_based_trajectory_arrow_positions(monkeypatch):
    arrow_tips = []

    def record_arrow(start_point, end_point, *args, **kwargs):
        arrow_tips.append(np.asarray(end_point))
        return []

    monkeypatch.setattr(drawer, "_draw_arrow_2d", record_arrow)
    fig, _ = drawer.phase_portrait(
        examples.games.standard_rps,
        starts=[],
        show_edge_flow=True,
        trajectory_arrows=[0.001],
        show_speed=False,
        show_equilibria=False,
    )

    expected_midpoints = np.array([
        drawer.simplex_to_plane_2p3s(0.5, 0.5),
        drawer.simplex_to_plane_2p3s(0.5, 0.0),
        drawer.simplex_to_plane_2p3s(0.0, 0.5),
    ])
    np.testing.assert_allclose(arrow_tips, expected_midpoints)
    plt.close(fig)


def test_2pop2s_default_edge_flow_draws_one_arrow_per_square_edge():
    fig, ax = drawer.phase_portrait(
        examples.games.matching_pennies,
        starts=[],
        show_edge_flow=True,
        trajectory_arrows=None,
        show_speed=False,
        show_equilibria=False,
    )

    heads = [artist for artist in ax.collections if isinstance(artist, PatchCollection)]
    assert len(heads) == 4
    assert all(not head.get_clip_on() for head in heads)
    plt.close(fig)


def test_hawk_dove_examples_distinguish_population_models():
    assert examples.games.hawk_dove.game_class == "1Pop2S"
    assert examples.games.two_population_hawk_dove.game_class == "2Pop2S"


def test_widget_combines_random_starts_with_edge_flow(monkeypatch):
    from pynamo_egt import interactive

    captured = {}
    fig, ax = plt.subplots()

    def record_phase_portrait(**kwargs):
        captured.update(kwargs)
        return fig, ax

    monkeypatch.setattr(interactive.drawer, "phase_portrait", record_phase_portrait)
    interactive._plot(
        game_key="2Pop2S",
        example_name="matching_pennies",
        num_traj=3,
        tmax=20,
        seed=7,
        show_speed=True,
        show_vector_field=False,
        show_trajectory_arrow=True,
        show_equilibria=True,
    )

    assert captured["show_edge_flow"] is True
    assert len(captured["starts"]) == 3
    np.testing.assert_allclose(
        captured["starts"],
        np.random.default_rng(7).random((3, 2)),
    )
    plt.close(fig)


def test_widget_figure_can_be_detached_without_closing_its_canvas():
    from matplotlib._pylab_helpers import Gcf
    from pynamo_egt import interactive

    fig = plt.figure()
    manager = fig.canvas.manager
    assert Gcf.figs.get(manager.num) is manager

    interactive.detach_widget_figure(figure=fig)

    assert Gcf.figs.get(manager.num) is None
    fig.canvas.draw()
    manager.destroy()


def test_draw_state_space_uses_supplied_axes_when_another_axes_is_current():
    fig, (target_ax, current_ax) = plt.subplots(ncols=2)
    plt.sca(current_ax)

    artists = drawer.draw_state_space(
        strategy_labels=["A", "B"],
        payoff_data=(np.zeros((2, 2)), np.zeros((2, 2))),
        ax=target_ax,
        font_size=13,
        zorder=20,
    )

    assert len(artists) == 4
    assert all(artist.axes is target_ax for artist in artists)
    assert len(target_ax.lines) == 4
    assert len(current_ax.lines) == 0
    plt.close(fig)


def test_phase_portrait_runs_for_all_supported_game_classes():
    for game_class in ("2Pop2S", "1Pop3S", "1Pop4S", "3Pop2S"):
        g = next(iter(examples.games.by_class(game_class).values()))
        fig, ax = drawer.phase_portrait(
            g,
            random_state=0,
            tmax=0.2,
            trajectory_arrows=[],
            show_speed=game_class in ("2Pop2S", "1Pop3S"),
            show_vector_field=True,
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)


def test_3pop2s_axes_derive_from_player_strategy_labels():
    payoff_tensors = tuple(np.zeros((2, 2, 2), dtype=float) for _ in range(3))
    custom_game = Game(
        "Custom three-population game",
        payoff_tensors,
        player_strategy_labels=[["Up", "Down"]] * 3,
        symmetric=False,
    )

    fig, ax = drawer.phase_portrait(
        custom_game,
        starts=[[0.5, 0.5, 0.5]],
        trajectory_arrows=[],
        show_equilibria=False,
        show_speed=False,
    )

    assert ax.get_xlabel() == "Population 1: Pr(Up)"
    assert ax.get_ylabel() == "Population 2: Pr(Up)"
    assert ax.get_zlabel() == "Population 3: Pr(Up)"
    plt.close(fig)


def test_phase_portrait_accepts_one_color_per_trajectory():
    g = examples.games.matching_pennies
    fig, ax = drawer.phase_portrait(
        g,
        starts=[[0.2, 0.7], [0.8, 0.3]],
        trajectory_color=["tab:blue", "tab:orange"],
        tmax=0.2,
        trajectory_arrows=[],
    )

    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_vector_field_only():
    g = examples.games.two_population_hawk_dove
    fig, ax = drawer.phase_portrait(
        g,
        show_trajectories=False,
        show_equilibria=False,
        show_speed=False,
        show_vector_field=True,
        vector_margin=0.0,
    )

    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_phase_portrait_accepts_equilibrium_edgecolor():
    g = examples.games.good_rps
    fig, ax = drawer.phase_portrait(
        g,
        show_trajectories=False,
        equilibrium_edgecolor="firebrick",
    )

    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_phase_portrait_rejects_unsupported_game_size():
    with pytest.raises(ValueError, match=r"shape \(5, 5\).*Supported game classes"):
        drawer.phase_portrait(Game("Five strategies", np.eye(5)))
