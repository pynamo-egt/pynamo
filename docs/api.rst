API reference
=============

.. currentmodule:: pynamo_egt

Games
-----

.. autosummary::
   :toctree: generated

   Game

Phase portraits
---------------

.. autosummary::
   :toctree: generated

   phase_portrait
   bifurcation_diagram
   phase_portrait_html

Equilibrium analysis
--------------------

.. autosummary::
   :toctree: generated

   analyze_equilibria
   equilibrium_table
   rest_points_nash
   rest_points_strict_nash
   rest_points_ess

Engine (preview)
----------------

The 1.0 engine, opt-in: build a backend-neutral scene, then draw it.
Currently only one-population three-strategy games are supported.

.. autosummary::
   :toctree: generated

   scene
   Replicator
   StateSpace
   render.matplotlib

Modules
-------

.. autosummary::
   :toctree: generated
   :recursive:

   game
   examples
   dynamics
   analysis
   drawer
   interactive
   render
