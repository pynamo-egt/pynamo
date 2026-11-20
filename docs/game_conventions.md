# Defining games

Build one-population games with `Game.symmetric` and two-population games with
`Game.bimatrix`:

```python
import pynamo_egt as pn

rps = pn.Game.symmetric(
    [[0, -1, 1], [1, 0, -1], [-1, 1, 0]], ["R", "P", "S"], name="Rock-Paper-Scissors"
)
```

## Textbook and focal-player layouts

Take the Prisoner's Dilemma, with strategies C (cooperate) and D (defect) for
both players. The two layouts store player 2's payoffs differently:

| | Textbook layout (`Game.bimatrix`) | Focal-player layout (legacy `Game`) |
|---|---|---|
| Rows of player 1's matrix | player 1's strategies | player 1's strategies |
| Rows of player 2's matrix | player 1's strategies | player 2's strategies |
| Player 1's matrix | $A = \begin{pmatrix} 3 & 0 \\ 5 & 1 \end{pmatrix}$ | $A = \begin{pmatrix} 3 & 0 \\ 5 & 1 \end{pmatrix}$ |
| Player 2's matrix | $B = \begin{pmatrix} 3 & 5 \\ 0 & 1 \end{pmatrix}$ | $B^\top = \begin{pmatrix} 3 & 0 \\ 5 & 1 \end{pmatrix}$ |
| Code | `pn.Game.bimatrix(A, B)` | `pn.Game("PD", (A, B.T))` |

In the textbook layout, $A_{ij}$ and $B_{ij}$ are the payoffs to players 1 and
2 when player 1 plays $i$ and player 2 plays $j$. In the focal-player layout,
each matrix has the receiving player's strategies as rows. Both games below are
identical:

```python
import numpy as np

A = np.array([[3, 0], [5, 1]])
B = np.array([[3, 5], [0, 1]])

textbook = pn.Game.bimatrix(A, B, [["C", "D"], ["C", "D"]], name="PD")
focal = pn.Game("PD", (A, B.T), strategy_labels=["C", "D"])
```

`Game.bimatrix` stores the focal-player form, so `textbook.matrices` is
`(A, B.T)`.

## Populations and payoffs

Every game exposes its strategy counts and labels per population:

```python
textbook.populations  # (2, 2)
textbook.labels       # (('C', 'D'), ('C', 'D'))
textbook.is_linear    # True
```

`payoffs(x)` takes one state per population in full coordinates, each an array
of shape `(..., n_p)`, and returns the payoff of every pure strategy with the
same batch shape:

```python
x = np.array([[1.0, 0.0], [0.5, 0.5]])  # two states of population 1
y = np.array([[0.0, 1.0], [0.5, 0.5]])  # two states of population 2
u1, u2 = textbook.payoffs([x, y])        # each of shape (2, 2)
```
