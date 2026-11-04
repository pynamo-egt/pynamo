"""Notebook widgets for exploring pyNamo replicator dynamics."""
from __future__ import annotations

import html as html_lib
from typing import List
import warnings

import matplotlib.pyplot as plt
import numpy as np

from . import analysis
from . import drawer
from . import examples

__all__ = ["detach_widget_figure", "launch_replicator_widget"]

try:  # pragma: no cover - optional dependency
    import ipywidgets as widgets
    from IPython.display import clear_output, display
except ImportError as exc:  # pragma: no cover - optional path
    widgets = None  # type: ignore[assignment]
    _IPYWIDGETS_IMPORT_ERROR = exc
else:
    _IPYWIDGETS_IMPORT_ERROR = None


GAME_LABELS = {
    "1Pop2S": "One population / 2 strategies",
    "1Pop3S": "One population / 3 strategies",
    "1Pop4S": "One population / 4 strategies",
    "2Pop2S": "Two populations / 2 strategies each",
    "3Pop2S": "Three populations / 2 strategies each",
}


def _sample_initial_conditions(game, count: int, rng: np.random.Generator) -> List[List[float]]:
    game_class = game.game_class
    if game_class in ("1Pop3S", "1Pop4S"):
        n = game.num_strategies()
        samples = rng.dirichlet(np.ones(n), size=count)
        return [vec[:-1].tolist() for vec in samples]

    if game_class == "2Pop2S":
        return rng.random((count, 2)).tolist()

    if game_class == "3Pop2S":
        return rng.random((count, 3)).tolist()

    return rng.random((count, 2)).tolist()


def _plot(
    game_key: str,
    example_name: str,
    *,
    fig=None,
    num_traj: int,
    tmax: float,
    seed: int,
    show_speed: bool,
    show_vector_field: bool,
    show_trajectory_arrow: bool,
    show_equilibria: bool,
):
    game = examples.games(example_name)

    rng = np.random.default_rng(seed)
    starts = None if game_key == "1Pop2S" else _sample_initial_conditions(game, num_traj, rng)
    trajectory_arrows = (
        None
        if game_key == "1Pop2S" and show_trajectory_arrow
        else [0.001]
        if show_trajectory_arrow
        else []
    )
    trajectory_color = "royalblue" if game_key in ("1Pop4S", "3Pop2S") else "black"
    fig, ax = drawer.phase_portrait(
        game=game,
        fig=fig,
        starts=starts,
        tmax=tmax,
        trajectory_arrows=trajectory_arrows,
        trajectory_color="black" if game_key in ("1Pop3S", "2Pop2S") else trajectory_color,
        trajectory_linewidth=1.2,
        show_edge_flow=game_key in ("1Pop3S", "2Pop2S"),
        show_speed=show_speed,
        speed_cmap=plt.cm.Spectral_r,
        speed_levels=80,
        show_vector_field=show_vector_field,
        vector_color="black" if game_key in ("1Pop3S", "2Pop2S") else "black",
        show_equilibria=show_equilibria,
        sink_color="black",
        saddle_color="gray",
        source_color="white",
        equilibrium_size=60,
        show_faces=True
    )
    ax.set_title(f"{game_key} - {game.name}", pad=18)
    return fig, ax


def _info_outputs(game, show_payoff_data: bool, show_analysis_table: bool) -> tuple[dict, ...]:
    outputs = []
    if show_payoff_data:
        outputs.append(_markdown_output("<br><h3>Payoff Data</h3><br>"))
        outputs.extend(_payoff_data_outputs(game))
    if show_analysis_table:
        outputs.append(_markdown_output("<br><h3>Equilibrium And Stability Analysis</h3><br>"))
        rows = analysis.analyze_equilibria(game).to_rows()
        outputs.append(_math_output(_analysis_table_latex(rows)))
    return tuple(outputs)


def _payoff_data_outputs(game) -> tuple[dict, ...]:
    payoff = game.payoff_data
    game_class = game.game_class

    if game_class in ("1Pop2S", "1Pop3S", "1Pop4S"):
        return (_math_output(_symmetric_payoff_latex(payoff, game.strategy_labels)),)

    if game_class == "2Pop2S":
        return (_math_output(_bimatrix_payoff_latex(payoff, game.player_strategy_labels)),)

    if game_class == "3Pop2S":
        return (_html_output(_three_player_payoff_html(payoff)),)

    return (_markdown_output(f"```\n{payoff}\n```"),)


def _markdown_output(markdown: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {"text/markdown": markdown},
        "metadata": {},
    }


def _html_output(html: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {"text/html": html},
        "metadata": {},
    }


def _math_output(latex: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {"text/latex": f"$\\displaystyle\n{latex.strip()}\n$"},
        "metadata": {},
    }


def _warning_outputs(caught_warnings) -> tuple[dict, ...]:
    if not caught_warnings:
        return ()

    lines = []
    seen = set()
    for warning in caught_warnings:
        category = warning.category.__name__
        message = str(warning.message)
        key = (category, message)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- **{category}**: {message}")

    if not lines:
        return ()

    return (_markdown_output("<br><h3>Warnings</h3><br>\n\n" + "\n".join(lines)),)


def detach_widget_figure(figure) -> None:
    """Keep a live widget figure active through later pyplot backend changes.

    Call this after displaying an ``ipympl`` figure when subsequent notebook
    cells switch to an inline backend or close pyplot-managed figures.
    """
    from matplotlib._pylab_helpers import Gcf

    manager = getattr(figure.canvas, "manager", None)
    if manager is not None and Gcf.figs.get(manager.num) is manager:
        Gcf.figs.pop(manager.num)


def _analysis_table_latex(rows: List[dict]) -> str:
    if not rows:
        return r"\text{No isolated equilibria found.}"

    columns = [
        ("Position", "Position"),
        ("Stability", "Stability Status"),
        ("Nash", "Nash"),
        ("ESS", "ESS"),
        ("Strict Nash", "Strict Nash"),
        ("Eigenvalues", "Eigenvalues"),
        ("Eigenvectors", "Eigenvectors"),
    ]

    header = " & ".join(rf"\text{{{label}}}" for label, _ in columns) + r" \\"
    table_rows = [header, r"\hline"]
    for row in rows:
        entries = [_analysis_cell(row[key]) for _, key in columns]
        table_rows.append(" & ".join(entries) + r" \\")

    return _latex_array("c" * len(columns), table_rows)


def _analysis_cell(value) -> str:
    if isinstance(value, list):
        if value and all(isinstance(item, list) for item in value):
            return _latex_vector_list(value)
        return _latex_column_vector(value)
    if value is None:
        return r"\text{None}"
    if isinstance(value, (bool, np.bool_)):
        return rf"\text{{{bool(value)}}}"
    if isinstance(value, str):
        return rf"\text{{{value}}}"
    return _format_number_latex(value)


def _latex_column_vector(values) -> str:
    entries = r" \\ ".join(_format_number_latex(value) for value in values)
    return rf"\begin{{pmatrix}} {entries} \end{{pmatrix}}"


def _latex_vector_list(vectors) -> str:
    entries = ", ".join(_latex_column_vector(vector) for vector in vectors)
    return rf"\left\{{ {entries} \right\}}"


def _format_number_latex(value) -> str:
    value = complex(value)
    real = float(value.real)
    imag = float(value.imag)

    if np.isclose(imag, 0.0):
        return _format_payoff(real)
    if np.isclose(real, 0.0):
        return rf"{_format_payoff(imag)}i"

    sign = "+" if imag > 0 else "-"
    return rf"{_format_payoff(real)} {sign} {_format_payoff(abs(imag))}i"


def _symmetric_payoff_latex(payoff, labels: List[str]) -> str:
    n = payoff.shape[0]
    labels = _strategy_labels_or_default(labels, n)
    header = " & " + " & ".join(labels) + r" \\"
    rows = [header, r"\hline"]
    for i, label in enumerate(labels):
        entries = " & ".join(_format_payoff(payoff[i, j]) for j in range(n))
        rows.append(rf"{label} & {entries} \\")
    return _latex_array("c|" + "c" * n, rows)


def _bimatrix_payoff_latex(payoff, player_strategy_labels: List[List[str]]) -> str:
    p1_payoff, p2_payoff = payoff
    row_labels = _strategy_labels_or_default(player_strategy_labels[0], 2)
    column_labels = _strategy_labels_or_default(player_strategy_labels[1], 2)
    rows = [
        " & " + " & ".join(column_labels) + r" \\",
        r"\hline",
    ]
    for i, row_label in enumerate(row_labels):
        entries = []
        for j in range(2):
            # Player 2 matrices use own action as row internally, so transpose for display.
            entries.append(
                rf"({_format_payoff(p1_payoff[i, j])}, {_format_payoff(p2_payoff[j, i])})"
            )
        rows.append(rf"{row_label} & {' & '.join(entries)} \\")
    return _latex_array("c|cc", rows)


def _three_player_payoff_html(payoff) -> str:
    tables = []
    for k in range(2):
        rows = []
        for i in range(2):
            cells = []
            for j in range(2):
                payoff_tuple = ", ".join(
                    _format_payoff(tensor[i, j, k]) for tensor in payoff
                )
                cells.append(f"({html_lib.escape(payoff_tuple)})")
            rows.append(
                f"""
                <tr>
                    <th>Player 1: {i + 1}</th>
                    <td>{cells[0]}</td>
                    <td>{cells[1]}</td>
                </tr>
                """
            )

        tables.append(
            f"""
            <div class="pynamo-3p2s-payoff-table" style="display: inline-block; margin-right: 3.5em; vertical-align: top;">
                <div style="text-align: center; font-weight: 600; margin-bottom: 0.35em;">
                    Player 3: {k + 1}
                </div>
                <table style="border-collapse: collapse; font-size: 1.05em;">
                    <thead>
                        <tr>
                            <th style="border-bottom: 1px solid black; padding: 0.25em 0.7em;"></th>
                            <th style="border-bottom: 1px solid black; padding: 0.25em 0.7em;">Player 2: 1</th>
                            <th style="border-bottom: 1px solid black; padding: 0.25em 0.7em;">Player 2: 2</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows)}
                    </tbody>
                </table>
            </div>
            """
        )

    return f"""
    <div style="display: flex; flex-wrap: wrap; gap: 2em; align-items: flex-start;">
        {''.join(tables)}
    </div>
    <style>
        .pynamo-3p2s-payoff-table th {{
            font-weight: 500;
            text-align: right;
            white-space: nowrap;
            padding: 0.25em 0.7em;
        }}
        .pynamo-3p2s-payoff-table td {{
            text-align: center;
            white-space: nowrap;
            padding: 0.25em 0.7em;
        }}
    </style>
    """


def _latex_array(column_spec: str, rows: List[str], display_math: bool = True) -> str:
    body = "\n".join(rows)
    array = rf"""
\begin{{array}}{{{column_spec}}}
{body}
\end{{array}}
"""
    if display_math:
        return array
    return array


def _strategy_labels_or_default(labels: List[str], n: int) -> List[str]:
    if len(labels) == n:
        return [_latex_label(label) for label in labels]
    return [str(idx + 1) for idx in range(n)]


def _latex_label(label: str) -> str:
    label = str(label).strip()
    if label.startswith("$") and label.endswith("$") and len(label) >= 2:
        return label[1:-1]
    return rf"\text{{{_escape_latex_text(label)}}}"


def _escape_latex_text(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _format_payoff(value) -> str:
    value = float(value)
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.3g}"


def launch_replicator_widget() -> None:
    """Display the interactive pyNamo widget inside a Jupyter notebook.

    The widget exposes the built-in example catalogue, trajectory controls,
    speed-field and vector-field toggles, payoff display, and equilibrium
    analysis display.

    Raises
    ------
    ImportError
        Raised when `ipywidgets` or IPython display tools are unavailable.

    Notes
    -----
    This function is intended for notebook use. In a development notebook, reload
    `interactive`, `drawer`, `analysis`, `dynamics`, `game`, and `examples`
    before calling it if those modules have been edited during the session.
    """
    if widgets is None:  # pragma: no cover
        raise ImportError(
            "ipywidgets is required for this feature"
        ) from _IPYWIDGETS_IMPORT_ERROR

    label_style = {"description_width": "140px"}
    wide_layout = widgets.Layout(width="720px")
    control_layout = widgets.Layout(width="420px")
    checkbox_layout = widgets.Layout(width="260px")

    game_dropdown = widgets.Dropdown(
        options=[
            (label, key)
            for key, label in GAME_LABELS.items()
            if examples.games.by_class(key)
        ],
        description="Game",
        value="1Pop3S",
        style=label_style,
        layout=wide_layout,
    )

    example_dropdown = widgets.Dropdown(
        description="Example",
        style=label_style,
        layout=wide_layout,
    )

    tmax_slider = widgets.FloatSlider(
        value=30,
        min=1,
        max=100,
        step=1,
        description="tmax",
        continuous_update=False,
        style=label_style,
        layout=control_layout,
    )
    traj_slider = widgets.IntSlider(
        value=3,
        min=1,
        max=12,
        step=1,
        description="Number of trajectories",
        continuous_update=False,
        style=label_style,
        layout=control_layout,
    )
    seed_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=999,
        step=1,
        description="Random seed",
        continuous_update=False,
        style=label_style,
        layout=control_layout,
    )
    speed_toggle = widgets.Checkbox(
        value=True,
        description="Show speed field",
        layout=checkbox_layout,
    )
    vector_toggle = widgets.Checkbox(
        value=False,
        description="Show vector field",
        layout=checkbox_layout,
    )
    eq_toggle = widgets.Checkbox(
        value=True,
        description="Show equilibria",
        layout=checkbox_layout,
    )
    payoff_toggle = widgets.Checkbox(
        value=True,
        description="Show payoff data",
        layout=checkbox_layout,
    )
    analysis_toggle = widgets.Checkbox(
        value=True,
        description="Show analysis table",
        layout=checkbox_layout,
    )
    arrow_toggle = widgets.Checkbox(
        value=True,
        description="Show trajectory arrow",
        layout=checkbox_layout,
    )

    updating_examples = False

    def _update_examples(*_):
        nonlocal updating_examples
        updating_examples = True
        try:
            games_by_class = examples.games.by_class(game_dropdown.value)
            options = [(g.name, name) for name, g in games_by_class.items()]
            example_dropdown.options = options
            if options:
                example_dropdown.value = options[0][1]
            one_dimensional = game_dropdown.value == "1Pop2S"
            for control in (
                traj_slider,
                tmax_slider,
                seed_slider,
                speed_toggle,
                vector_toggle,
            ):
                control.disabled = one_dimensional
        finally:
            updating_examples = False

    _update_examples()
    game_dropdown.observe(_update_examples, names="value")

    controls = widgets.VBox(
        [
            game_dropdown,
            example_dropdown,
            widgets.HBox([traj_slider, tmax_slider]),
            seed_slider,
            widgets.HBox([speed_toggle, vector_toggle, eq_toggle]),
            widgets.HBox([arrow_toggle, payoff_toggle, analysis_toggle]),
        ]
    )

    info_output = widgets.Output()
    fallback_plot_output = widgets.Output()
    plot_area = widgets.VBox([fallback_plot_output])

    last_render_state = None
    figure = None

    def _render(*args):
        nonlocal figure, last_render_state
        if updating_examples:
            return
        render_state = (
            game_dropdown.value,
            example_dropdown.value,
            traj_slider.value,
            tmax_slider.value,
            seed_slider.value,
            speed_toggle.value,
            vector_toggle.value,
            eq_toggle.value,
            payoff_toggle.value,
            analysis_toggle.value,
            arrow_toggle.value,
        )
        if render_state == last_render_state:
            return
        last_render_state = render_state

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", DeprecationWarning)
            current_game = examples.games(example_dropdown.value)
            info_outputs = list(
                _info_outputs(
                    current_game,
                    show_payoff_data=payoff_toggle.value,
                    show_analysis_table=analysis_toggle.value,
                )
            )

            if figure is not None:
                figure.clear()
            first_render = figure is None
            with plt.ioff():
                figure, _ = _plot(
                    game_dropdown.value,
                    example_dropdown.value,
                    fig=figure,
                    num_traj=traj_slider.value,
                    tmax=tmax_slider.value,
                    seed=seed_slider.value,
                    show_speed=speed_toggle.value,
                    show_vector_field=vector_toggle.value,
                    show_trajectory_arrow=arrow_toggle.value,
                    show_equilibria=eq_toggle.value,
                )

        if first_render:
            if isinstance(figure.canvas, widgets.Widget):
                plot_area.children = (figure.canvas,)
                detach_widget_figure(figure)
            else:
                plot_area.children = (fallback_plot_output,)
                with fallback_plot_output:
                    clear_output(wait=True)
                    display(figure)
        else:
            figure.canvas.draw_idle()

        info_outputs.extend(_warning_outputs(caught_warnings))
        info_output.outputs = tuple(info_outputs)

    for widget in [
        game_dropdown,
        example_dropdown,
        traj_slider,
        tmax_slider,
        seed_slider,
        arrow_toggle,
        speed_toggle,
        vector_toggle,
        eq_toggle,
        payoff_toggle,
        analysis_toggle,
    ]:
        widget.observe(_render, names="value")

    display(widgets.VBox([controls, info_output, plot_area]))
    _render()
