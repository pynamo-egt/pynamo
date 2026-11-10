import importlib.util

import pytest
import pynamo_egt as pn

requires_plotly = pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None,
    reason="plotly is not installed (install the 'plotly' extra)",
)


@requires_plotly
def test_phase_portrait_html_returns_html_string():
    game = pn.examples.games.rps_with_twin
    html = pn.phase_portrait_html(
        game,
        tmax=10,
        trajectory_number=2,
        show_speed=False,
        random_state=0,
    )
    assert isinstance(html, str)
    assert len(html) > 0
    assert "plotly" in html.lower()


@requires_plotly
def test_phase_portrait_html_with_speed():
    game = pn.examples.games.rps_with_twin
    html = pn.phase_portrait_html(
        game,
        tmax=5,
        trajectory_number=1,
        show_speed=True,
        random_state=0,
    )
    assert "plotly" in html.lower()


@requires_plotly
def test_phase_portrait_html_writes_file(tmp_path):
    game = pn.examples.games.rps_with_twin
    out = tmp_path / "portrait.html"
    html = pn.phase_portrait_html(
        game,
        path=str(out),
        tmax=5,
        trajectory_number=1,
        show_speed=False,
        random_state=0,
    )
    assert out.exists()
    assert out.read_text() == html


@requires_plotly
def test_phase_portrait_html_3pop2s():
    game = pn.examples.games.by_class("3Pop2S")
    name = next(iter(game))
    html = pn.phase_portrait_html(
        game[name],
        tmax=5,
        trajectory_number=2,
        show_speed=False,
        random_state=0,
    )
    assert isinstance(html, str)
    assert "plotly" in html.lower()


@requires_plotly
def test_phase_portrait_html_3pop2s_with_speed():
    game = pn.examples.games.by_class("3Pop2S")
    name = next(iter(game))
    html = pn.phase_portrait_html(
        game[name],
        tmax=5,
        trajectory_number=1,
        show_speed=True,
        random_state=0,
    )
    assert "plotly" in html.lower()


@requires_plotly
def test_phase_portrait_html_rejects_unsupported():
    game = pn.examples.games.good_rps
    with pytest.raises(ValueError, match="1Pop4S"):
        pn.phase_portrait_html(game)


def test_phase_portrait_html_raises_without_plotly(monkeypatch):
    import pynamo_egt.export as export_mod
    monkeypatch.setattr(export_mod, "_PLOTLY_AVAILABLE", False)
    monkeypatch.setattr(export_mod, "_PLOTLY_IMPORT_ERROR", ImportError("plotly not found"))
    with pytest.raises(ImportError, match="plotly"):
        export_mod.phase_portrait_html(pn.examples.games.rps_with_twin)
