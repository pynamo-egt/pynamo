"""
Cyclic mismatching pennies
==========================

An asymmetric three-population game on the cube. Trajectories starting near the
interior rest point spiral away from it towards a cycle on the boundary.
Bottom-right panel of Figure 1 in the paper.
"""

import matplotlib.pyplot as plt

import pynamo_egt as pn

fig, ax = pn.phase_portrait(
    game=pn.examples.games.cyclic_mismatching_pennies,
    figsize=(8, 8),
    view_elev=16,
    view_azim=127,
    show_faces=True,
    face_alpha=0.5,
    starts=[[0.52, 0.50, 0.48], [0.60, 0.50, 0.40], [0.70, 0.45, 0.35], [0.20, 0.65, 0.75]],
    tmax=400,
    trajectory_arrows=[],
    trajectory_color=["royalblue", "firebrick", "goldenrod", "darkgreen"],
    trajectory_linewidth=1.5,
    equilibrium_size=25,
    simplex_font_size=18,
)
ax.set_title("")
ax.set_box_aspect((1, 1, 1), zoom=1.1)
plt.show()
