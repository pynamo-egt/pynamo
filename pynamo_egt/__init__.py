"""Replicator dynamics, analysis, and plotting for evolutionary games."""
from importlib import import_module

from . import analysis, drawer, dynamics, examples, game
from .game import Game
from .drawer import phase_portrait, bifurcation_diagram
from .analysis import (
    analyze_equilibria, equilibrium_table,
    rest_points_nash, rest_points_strict_nash, rest_points_ess,
)
from .export import phase_portrait_html

__version__ = "0.4.0"
__all__ = [
    "Game", "phase_portrait", "bifurcation_diagram", "phase_portrait_html",
    "analyze_equilibria", "equilibrium_table",
    "rest_points_nash", "rest_points_strict_nash", "rest_points_ess",
    "analysis", "drawer", "dynamics", "examples", "game", "interactive",
]


def __getattr__(name):
    # Load the optional notebook interface only when requested.
    if name == "interactive":
        module = import_module(".interactive", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
