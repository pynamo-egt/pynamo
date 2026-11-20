import subprocess
import sys
import pynamo_egt as pn


def test_public_exports():
    assert pn.Game is pn.game.Game
    assert pn.phase_portrait is pn.drawer.phase_portrait
    for name in (
        'analyze_equilibria',
        'equilibrium_table',
        'rest_points_nash',
        'rest_points_strict_nash',
        'rest_points_ess',
    ):
        assert getattr(pn, name) is getattr(pn.analysis, name)
def test_core_import_without_notebook_dependencies():
    subprocess.run([sys.executable, '-c', '''
import sys
sys.modules['ipywidgets'] = None
sys.modules['IPython'] = None
import pynamo_egt as pn
assert 'pynamo_egt.interactive' not in sys.modules
assert pn.phase_portrait is pn.drawer.phase_portrait
assert pn.interactive is pn.interactive
assert callable(pn.interactive.launch_replicator_widget)
'''], check=True, capture_output=True, text=True)


def test_analysis_works_without_sympy():
    subprocess.run([sys.executable, '-c', '''
import sys
sys.modules['sympy'] = None
import pynamo_egt as pn
for game in (pn.examples.games.good_rps, pn.examples.games.matching_pennies,
             pn.examples.games.cyclic_mismatching_pennies):
    assert pn.analyze_equilibria(game).equilibria
'''], check=True, capture_output=True, text=True)
