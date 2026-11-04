"""
Phase line of a two-strategy game
=================================

A symmetric 2×2 game whose interior rest point is the only stable one: the
arrows point inwards and both pure states are unstable. Panel B of Figure 2 in
the paper.
"""

import matplotlib.pyplot as plt
import numpy as np

import pynamo_egt as pn

game = pn.Game(
    name="",
    payoffs=np.array([[0, 7], [13, 0]], dtype=float),
    strategy_labels=[r"$S_1$", r"$S_2$"],
)
fig, ax = pn.phase_portrait(
    game,
    trajectory_color="gray",
    trajectory_linewidth=2.5,
    sink_color="black",
    source_color="white",
    equilibrium_edgecolor="black",
    equilibrium_size=160,
    simplex_font_size=24,
    arrow_size=0.08,
    arrow_width=0.03,
)
ax.set_title("")
plt.show()
