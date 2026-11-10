"""SymPy reference for replicator rest points, used as a differential oracle.

This reproduces how pyNamo computed rest points before the face-enumeration
solver: one call to SymPy ``solve`` on the whole replicator system.
"""

from sympy import Symbol, solve

from pynamo_egt import dynamics
from pynamo_egt.game import infer_game_class

_FIELDS = {
    "1Pop3S": (2, dynamics.replicator_2p3s),
    "1Pop4S": (3, dynamics.replicator_2p4s),
    "2Pop2S": (2, dynamics.replicator_2p2s),
    "3Pop2S": (3, dynamics.replicator_3p2s),
}


def sympy_rest_points(payoff_data) -> tuple[list[list[float]], bool]:
    """Return ``(points, isolated)`` in SymPy's order.

    ``isolated`` is False when SymPy reports a parametric (non-isolated) or
    non-real solution; such games have degenerate ties and are not compared.
    """
    dimension, field = _FIELDS[infer_game_class(payoff_data)]
    symbols = [Symbol(name) for name in "xyz"[:dimension]]
    equations = [eq for eq in field(symbols, 0, payoff_data) if eq != 0]
    if not equations:
        return [], False
    points = []
    for solution in solve(equations, *symbols, dict=True):
        try:
            points.append([float(solution[symbol]) for symbol in symbols])
        except (KeyError, TypeError):
            return [], False
    return points, True
