---
title: "pyNamo-EGT: Phase portraits of evolutionary game dynamics in Python"
tags:
  - Python
  - evolutionary game theory
  - replicator dynamics
  - game theory
  - visualization
  - Figures
authors:
  - name: Slimane Dridi
    orcid: 0000-0002-0159-9911
    affiliation: 1
  - name: Benjamin Giraudon
    affiliation: 2
affiliations:
  - name: Independent Researcher
    index: 1
  - name: Independent Researcher
    index: 2
date: 19 October 2026
bibliography: paper.bib
---

# Summary

We present `pyNamo-EGT`, a Python package for drawing phase portraits in evolutionary game theory. Its first version focuses on the replicator dynamics. It allows users to plot the phase portrait for the replicator dynamics of symmetric games (with up to four strategies) and asymmetric games (with up to three populations), producing publication-quality Matplotlib figures with trajectories, speed fields, vector fields, rest points, and stability markers (\autoref{fig:examples}).

The package covers the full analytical pipeline: defining a game by its payoff matrix (or matrices, for asymmetric games), computing the rest points of the replicator dynamics and classifying their stability, and rendering the phase portrait in a single high-level call. It ships with a curated catalogue of named example games (e.g., Rock–Paper–Scissors, Hawk–Dove, Matching Pennies, etc.) with descriptions, references, and parameter notes, and an interactive Jupyter widget for exploratory use. `pyNamo-EGT` is available on PyPI, with interactive tutorials on Binder and Google Colab, making it immediately accessible without a local installation.

![Phase portraits of the replicator dynamics for four games, produced by `pyNamo-EGT`: Rock–Paper–Scissors (top left) [@Sandholm2010], two-population Hawk–Dove (top right) [@MaynardSmithPrice1973; @HofbauerSigmund1998], four-strategy chaotic dynamics (bottom left) [@Arneodo1980], and asymmetric three-population cyclic mismatching pennies (bottom right) [@Sandholm2010]. In the 2D plots (top), a black dot is a sink, a gray-filled dot is a saddle, and a white-filled dot is a source. The color gradient in the 2D plots represents the speed of the replicator dynamics, from slow (dark) to fast (light). In the 3D plots (bottom), a black sphere is a sink, a dark gray sphere is a saddle, and a light gray sphere is a source. \label{fig:examples}](gallery.pdf){width=100%}


# Statement of need

 Evolutionary game theory studies how populations of agents (be they humans, animals, plants, bacteria, computers on a network, etc.) adapt to each other when they socially interact [@Alexander2023; @Weibull1995; @HofbauerSigmund1998; @Sandholm2010]. The mathematical models in evolutionary game theory usually specify a base *game* that describes the rules and consequences of the outcome of a social interaction, a *matching process* that determines who interacts with whom, and an *updating rule* that tells how the result of a social interaction affects the behavior of agents in future interactions. Biologists, economists, social scientists, computer scientists, and researchers in other fields rely on evolutionary game theory as a tool to answer fundamental questions in their respective fields.
 
When analyzing an evolutionary game, one often deals with nonlinear ordinary differential equations (ODEs) that can produce complex behavior. These equations may have multiple rest points, with different stability properties, and sometimes display limit cycles, or even chaos in higher dimensions [@Arneodo1980; @Schnabl1991; @Skyrms1992], as illustrated in \autoref{fig:examples}. One way to explore and better understand these dynamics is to plot the phase portrait associated with them. The phase portrait is a graphical representation of an ODE's dynamics on its state space, typically showing trajectories, rest points, and their stability. Researchers and teachers would benefit greatly from having easy access to a free and open-source general-purpose library that allows one to graphically represent evolutionary game dynamics in all the visualizable classes of games.
 
Such a tool should also do all the necessary rigorous work in the background. In evolutionary game theory, the rigorous work mainly consists in finding rest points and analyzing their stability. Many cases are amenable to an almost automatic treatment because the linear stability analysis (which is well-suited to automation) often provides a relatively complete picture of the phase portrait, at least for planar state spaces (2D) [@HofbauerSigmund1998; @Weibull1995]. With `pyNamo-EGT`, we strive to provide this automated treatment of evolutionary games. To our knowledge, `pyNamo-EGT` is the only open-source Python package that covers all visualizable game classes with automated stability analysis and publication-quality output.
 
Evolutionary game dynamics occur in the simplex or product of simplices. For a game with $n$ strategies, the population state is a vector $x = (x_1, \ldots, x_n)$ of strategy frequencies, where $x_i \geq 0$ and $\sum_i x_i = 1$. The space of possible frequency vectors $x$ is a geometric object called the $(n-1)$-simplex, denoted $\Delta_{n-1}$. In \autoref{fig:simplex} below, we show how the $1$-simplex, $\Delta_1$, is embedded in  $\mathbb{R}^2$. (For multi-population games, the state lives in a product of such simplices, one per population.)

The replicator dynamics [@TaylorJonker1978] then take the form
\begin{equation}
\dot{x}_i = x_i(f_i(x) - \bar{f}(x)),
\label{eq:replicator}
\end{equation}
where $f_i(x)$ is the fitness of strategy $i$ and $\bar{f}(x)$ is the population mean fitness [@HofbauerSigmund1998]. Even though $x \in \mathbb{R}^n$, the constraint $\sum_i x_i = 1$ restricts the state space to $\Delta_{n-1}$, which is an $(n-1)$-dimensional subset of $\mathbb{R}^n$. Analysis of rest point stability has to be restricted as well to this relevant state space. Hence one cannot rely solely on standard ODE solvers and packages, as those would consider deviations in all of $\mathbb{R}^n$ (\autoref{fig:simplex}).

![(A) The $\Delta_1$ simplex as a subset of $\mathbb{R}^2$. If we consider a two-player, two-strategy game with strategies called $S_1$ and $S_2$, and the respective frequencies of these strategies $x_1$ and $x_2$, the simplex is the line segment in $\mathbb{R}^2$ where $x_2=1-x_1$, and both $x_1 \geq 0$ and $x_2 \geq 0$. For the replicator dynamics to stay in the simplex (relevant for stability analysis), the population can only move along admissible directions, shown in green. Taking a direction in a non-admissible direction, in orange, would lead the population outside of the simplex and the state vector would no longer describe frequencies, i.e., proportions. At each boundary endpoint, only one direction along the simplex is admissible (green, pointing inward); the opposite direction (orange) would drive a strategy frequency below 0 or above 1, leaving the state space. (B) The projection of the simplex $\Delta_1$ onto $\mathbb{R}$, produced with `pyNamo-EGT`. Note that we used a payoff matrix that makes the interior rest point (shown in black) the only stable one, as indicated by the arrows pointing inwards. The two white-filled boundary rest points are unstable. \label{fig:simplex}](simplex_figure_AB.pdf){width=70%}


# State of the field

Dynamo [@Sandholm2010; @Sandholm2012] is the closest relative to our package for plotting and analyzing deterministic evolutionary game dynamics. Its design inspired ours. It is a Mathematica package, written in the Wolfram Language. It allows one to draw trajectories inside the simplex for every symmetric and asymmetric game that can be visualized. It covers not only the replicator dynamics but other population dynamics, such as best-response, logit, etc. For a combination of game and evolutionary dynamic, the package can generate a report with the phase portrait (trajectories or vector field) as well as the rest points of the dynamic, together with their stability class.

In the Python ecosystem, EGTtools [@Fernandez2023] is a comprehensive library for both deterministic and stochastic evolutionary dynamics. For deterministic dynamics, it covers the replicator dynamics for games with up to 3 strategies. For finite populations, it implements stochastic processes such as the Moran process and provides tools for fixation probabilities and invasion diagrams, with a C++ backend for efficient simulations. EGTtools complements `pyNamo-EGT` for stochastic and finite-population analyses; the two packages share a common payoff-matrix convention and can be used together.

We also mention Nashpy [@Knight2018] that provides tools both for finding static equilibria and evolutionary dynamics analysis. Its functions for evolutionary dynamics are restricted to 2-player games. They include not only the replicator dynamics but other dynamics such as fictitious play, stochastic fictitious play and introspection dynamics as well as stochastic models. The output is time series in the form of numpy arrays rather than plots or phase portraits.

OpenSpiel [@Lanctot2019] is a large reinforcement learning framework whose `egt` module provides replicator dynamics and two Matplotlib projections: a triangle plot for one-population 3-strategy games and a square plot for two-population 2×2 games. Games with more strategies or more populations are handled by the dynamics classes but have no corresponding visualization, and phase lines for one-population 2-strategy games (\autoref{fig:simplex}B) are absent. Rest-point computation, stability classification, and Nash or ESS checks are not provided. The module is peripheral to OpenSpiel's main focus (extensive-form games and RL algorithms) and is not covered in its documentation.

Finally, there exists an older unmaintained package called egtplot [@egtplot] that only allows one to plot 3-strategy replicator dynamics in one population.

# Software design

Our package was designed to make the path from a mathematical model, expressed in the language of evolutionary game theory, to a phase portrait as short as possible. This guiding principle has shaped most of our design decisions, which are centered around two constructs: (1) the `Game` class; (2) the `phase_portrait()` function. 

Our `Game` class directly translates the idea that a game in the mathematical sense is a collection of players, strategies, and payoffs. Rather than passing raw payoff arrays to each function call, the user defines the game once, bundling payoff data together with all the information needed to analyse and visualise it: the number of players, the number of strategies per player, and their labels. From that point on, a single call to `phase_portrait(game=my_game)` is all that is needed to produce a phase portrait. Beyond this convenience, the structure of the payoff data lets `pyNamo-EGT` automatically determine what kind of game it is dealing with, removing a category of errors that would arise if the user had to declare this separately. Strategy and player labels flow directly into the figure without additional arguments. Finally, the `Game` class also lays the foundation for future extensions: all of the current games implemented in `pyNamo-EGT` are only normal-form games (with linear payoff functions), but contributors wishing to add support for nonlinear payoff functions can subclass it without changing the rest of the API.

Assuming the user has defined a game object in `my_game`, drawing an actual phase portrait may be as simple as
```python
import pynamo_egt as pn
fig, ax = pn.phase_portrait(game=my_game)
```
Note, however, that the user does not need to understand the `Game` class before being able to use `phase_portrait()`. They could just pick a game from our predefined list of examples, which includes many of the well-known models from the evolutionary game theory literature such as the Hawk-Dove game [@MaynardSmithPrice1973; @MaynardSmithParker1976]: 
```python
import pynamo_egt as pn
fig, ax = pn.phase_portrait(game=pn.examples.games.hawk_dove)
```

As mentioned above, internally, `pyNamo-EGT` infers the game class (number of populations and strategies, game symmetry) from the properties of the game object passed to `phase_portrait()` so that it can use the appropriate state space. Our package covers five possible game classes, as outlined in \autoref{tab:game_classes}.

| Class name in `pyNamo-EGT` | Populations and strategies             | State space |
| -------------------------- | -------------------------------------- | ----------- |
| `1Pop2S`                   | One population, two strategies         | Line        |
| `1Pop3S`                   | One population, three strategies       | Triangle    |
| `1Pop4S`                   | One population, four strategies        | Tetrahedron |
| `2Pop2S`                   | Two populations, two strategies each   | Unit square |
| `3Pop2S`                   | Three populations, two strategies each | Unit cube   |

Table: Description of the five game classes supported by `pyNamo-EGT`. \label{tab:game_classes}

In all one-population classes, interactions are assumed to be pairwise and symmetric; generalizing to $N$-player single-population games would actually generate a nonlinear $f_i(x)$ in equation \eqref{eq:replicator} [@Pena2014], which is left for a future version.

The output of `phase_portrait()` is a Matplotlib figure object, which can be further customized using standard Matplotlib calls. However, knowing Matplotlib is not a prerequisite. The `phase_portrait()` function exposes its own parameters for the most common customizations (colors, line widths, arrow sizes, fonts, and figure dimensions) so that most users never need to interact with Matplotlib directly.

In the Statement of need section, we mentioned restricting perturbations to admissible directions, when analyzing rest point stability. This is achieved differently for interior vs. boundary rest points. For interior rest points, it suffices to treat points of the simplex as elements of $\mathbb{R}^{n-1}$ rather than $\mathbb{R}^n$ by dropping one coordinate, since coordinates sum to one; the Jacobian at a rest point is then of dimension $(n-1)\times (n-1)$, and all its eigendirections are automatically tangent to the simplex. For boundary rest points, an additional filter is applied: eigenvectors whose direction would drive any strategy frequency outside $[0,1]$ are discarded before the stability classification is made, as illustrated in \autoref{fig:simplex}.

We finally mention how we deal with limitations of the automatic rest point detection and linear stability analysis. While a significant number of cases can be treated automatically by `pyNamo-EGT`, certain games may display degenerate equilibrium sets (lines or curves, or more complex manifolds in 3D), and we warn the user explicitly when this is the case. Our package detects when a set of rest points is more complex than isolated points but does not go on and compute the geometry of those manifolds. Also, sometimes linear stability analysis is insufficient to establish the stability properties of a rest point. Our package tries to exploit known results about the replicator dynamics [@HofbauerSigmund1998; @Weibull1995] when possible so it can go beyond linear stability analysis. When not possible, we warn users and provide the raw eigenvalue data so that they can investigate stability further.

# Research impact statement

Our package has been used in a scientific preprint on the evolution of behavioral plasticity by the first author [@Dridi2019] and has received more than 450 downloads from PyPI in the month of October 2026.

# AI usage disclosure

The package development started before the advent of AI, so the core of the code was written by the authors. The design philosophy and ideas for features were in place from the start, before we had access to agentic AI. More recently, AI agents (more specifically, ChatGPT's Codex and Claude Code) were used to help implement new ideas. AI was also used to help edit parts of this manuscript. All code was thoroughly reviewed, tested, and when necessary modified by the authors before integration in the package. All text produced with AI assistance was reviewed and edited by the authors before inclusion in this article.

# Acknowledgements

We would like to thank Jorge Peña for valuable comments on version 0.2.0 of the package.

# References
