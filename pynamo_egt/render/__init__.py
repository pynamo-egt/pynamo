"""Renderers that draw scenes built by :func:`pynamo_egt.scene`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["matplotlib"]


def matplotlib(
    scene: dict[str, Any], ax: Optional[Axes] = None, *, fig: Optional[Figure] = None, **style: Any
) -> tuple[Figure, Axes]:
    """Draw ``scene`` with Matplotlib and return ``(fig, ax)``.

    Parameters
    ----------
    scene : dict
        Scene from :func:`pynamo_egt.scene`.
    ax, fig : optional
        Axes (or figure) to draw on; a new figure by default.
    **style
        Overrides of the defaults used by :func:`pynamo_egt.phase_portrait`,
        e.g. ``trajectory_color="tab:blue"`` or ``equilibrium_size=60``.
    """
    from ._matplotlib import render

    return render(scene, ax=ax, fig=fig, **style)
