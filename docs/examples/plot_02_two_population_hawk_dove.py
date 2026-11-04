"""
Two-population Hawk–Dove
========================

Hawk–Dove between two populations on the square. The interior rest point is a
saddle and the two asymmetric conventions are sinks. Top-right panel of
Figure 1 in the paper.
"""

import matplotlib.pyplot as plt

import pynamo_egt as pn

fig, ax = pn.phase_portrait(
    game=pn.examples.games.two_population_hawk_dove,
    figsize=(8, 8),
    starts=[[0.25, 0.85], [0.35, 0.35], [0.9, 0.9], [0.85, 0.25], [0.85, 0.75], [0.75, 0.85]],
    show_speed=True,
    speed_levels=200,
    show_vector_field=True,
    vector_color="white",
    speed_cmap=plt.cm.cividis,
    trajectory_linewidth=1.5,
)
ax.set_title("")
plt.show()
