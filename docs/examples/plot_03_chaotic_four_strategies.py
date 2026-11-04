"""
Chaotic dynamics with four strategies
=====================================

A four-strategy game (Arneodo et al., 1980) whose replicator dynamics are
chaotic, drawn in the tetrahedron. Bottom-left panel of Figure 1 in the paper.
"""

import matplotlib.pyplot as plt

import pynamo_egt as pn

fig, ax = pn.phase_portrait(
    game=pn.examples.games.chaotic_four_strategy_game,
    figsize=(8, 8),
    view_elev=20,
    view_azim=-56,
    starts=[[0.22, 0.25, 0.25], [0.20, 0.20, 0.60], [0.45, 0.10, 0.20]],
    tmax=90,
    trajectory_step=0.02,
    trajectory_color="royalblue",
    trajectory_linewidth=1.2,
    trajectory_arrows=[0.001],
    show_speed=False,
    show_vector_field=False,
    show_faces=True,
    face_colors=["#666666", "#999999", "#CCCCCC", "none"],
    face_alpha=0.18,
    simplex_font_size=18,
    equilibrium_size=30,
    equilibrium_edgecolor="black",
)
ax.set_title("")
ax.set_box_aspect((1, 1, 1), zoom=1.2)
plt.show()
