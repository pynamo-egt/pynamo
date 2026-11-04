"""
Rock–paper–scissors
===================

A good Rock–Paper–Scissors game on the triangle: trajectories spiral into the
interior sink, the speed of the dynamics is shaded, and the edges show the flow
between the pure strategies. Top-left panel of Figure 1 in the paper.
"""

import matplotlib.pyplot as plt

import pynamo_egt as pn

fig, ax = pn.phase_portrait(
    game=pn.examples.games.good_rps,
    figsize=(8, 8),
    starts=[[0.15, 0.15], [0.15, 0.70], [0.70, 0.15]],
    trajectory_color="black",
    trajectory_linewidth=1.2,
    show_edge_flow=True,
    show_speed=True,
    speed_cmap=plt.cm.plasma_r,
    speed_grid=80,
    speed_levels=200,
    simplex_font_size=18,
    equilibrium_size=80,
)
ax.set_title("")
plt.show()
