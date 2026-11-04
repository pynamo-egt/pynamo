"""Exact replicator rest points by support (face) enumeration.

On each face of the state space, the replicator rest-point conditions are a
small linear system: every strategy in the support earns the same payoff.
Solving these systems in exact rationals finds every rest point (including
those outside the simplex, which callers filter) and detects continua
without a general-purpose symbolic solver, so analysis cannot hang.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Iterable, Iterator, Literal, Optional, Sequence

import numpy as np

Point = list[Fraction]
Matrix = list[list[Fraction]]
SolveKind = Literal["unique", "continuum", "none"]


def to_fraction(value: float) -> Fraction:
    """Read a payoff as an exact rational at 15 significant digits.

    This matches how SymPy reads floats, so for example 0.1 + 0.2 == 0.3.
    """
    return Fraction(format(float(value), ".15g"))


def solve_linear(
    matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]
) -> tuple[SolveKind, Optional[Point]]:
    """Solve a square system exactly by Gauss–Jordan elimination.

    Returns ``("unique", x)``, ``("continuum", None)`` or ``("none", None)``.
    """
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    size = len(matrix[0]) if matrix else 0
    pivot_columns: list[int] = []
    rank = 0
    for column in range(size):
        pivot = next((r for r in range(rank, len(rows)) if rows[r][column] != 0), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        lead = rows[rank][column]
        rows[rank] = [value / lead for value in rows[rank]]
        for r in range(len(rows)):
            if r != rank and rows[r][column] != 0:
                factor = rows[r][column]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[rank])]
        pivot_columns.append(column)
        rank += 1
    if any(row[-1] != 0 for row in rows[rank:]):
        return "none", None
    if rank < size:
        return "continuum", None
    solution = [Fraction(0)] * size
    for r, column in enumerate(pivot_columns):
        solution[column] = rows[r][-1]
    return "unique", solution


class FaceSystem:
    """Linear rest-point conditions on one face, over all coordinates.

    Coordinates fixed by the face (strategies outside the support) appear as
    equations too, so a singular consistent system is exactly a continuum of
    rest points lying in that face.
    """

    def __init__(self, matrix: Matrix, rhs: list[Fraction]) -> None:
        self.matrix = matrix
        self.rhs = rhs

    def contains(self, point: Point) -> bool:
        """Whether ``point`` satisfies every equation of this face."""
        return all(
            sum(a * x for a, x in zip(row, point)) == b
            for row, b in zip(self.matrix, self.rhs)
        )


def symmetric_faces(payoff_matrix: np.ndarray) -> Iterator[FaceSystem]:
    """Yield the rest-point system on every face of a symmetric game's simplex."""
    A = [[to_fraction(value) for value in row] for row in payoff_matrix]
    n = len(A)
    zero, one = Fraction(0), Fraction(1)
    for size in range(1, n + 1):
        for support in combinations(range(n), size):
            first = support[0]
            matrix = [[one if j == i else zero for j in range(n)] for i in range(n) if i not in support]
            matrix += [[A[i][j] - A[first][j] for j in range(n)] for i in support[1:]]
            matrix.append([one] * n)
            rhs = [zero] * (n - 1) + [one]
            yield FaceSystem(matrix, rhs)


def enumerate_rest_points(
    faces: Iterable[FaceSystem], vertices: Sequence[Point]
) -> tuple[list[Point], bool]:
    """Collect rest points face by face, preserving the legacy output contract.

    Returns ``(points, degenerate)``. ``points`` lists the isolated rest points
    in lexicographic order, then any vertex not already listed, in the order
    given. Non-vertex points lying on a continuum are dropped; vertices are
    always listed. ``degenerate`` is True when some face carries a continuum.
    """
    isolated: list[Point] = []
    continua: list[FaceSystem] = []
    for face in faces:
        kind, point = solve_linear(face.matrix, face.rhs)
        if kind == "continuum":
            continua.append(face)
        elif point is not None and point not in isolated:
            isolated.append(point)

    isolated = sorted(
        point for point in isolated if not any(face.contains(point) for face in continua)
    )
    for vertex in vertices:
        if vertex not in isolated:
            isolated.append(vertex)
    return isolated, bool(continua)


def symmetric_rest_points(payoff_matrix: np.ndarray) -> tuple[list[Point], bool]:
    """Rest points of a symmetric game in full simplex coordinates."""
    n = len(payoff_matrix)
    vertices = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    return enumerate_rest_points(symmetric_faces(payoff_matrix), vertices)


def _indifference(payoff_matrix):
    """Coefficients (slope, intercept) of u(first) - u(second) in the opponent's
    probability of their first strategy, for a focal-player 2×2 matrix."""
    (a, b), (c, d) = [[to_fraction(value) for value in row] for row in payoff_matrix]
    return a - b - c + d, b - d


def bimatrix_faces(payoff_data):
    """Yield the rest-point system on every face of a 2Pop2S square.

    Each population either plays a pure strategy (its coordinate is 0 or 1)
    or is indifferent, which is linear in the other population's coordinate.
    """
    zero, one = Fraction(0), Fraction(1)
    (alpha, beta), (gamma, delta) = (_indifference(matrix) for matrix in payoff_data)
    x_options = [([one, zero], zero), ([one, zero], one), ([zero, alpha], -beta)]
    y_options = [([zero, one], zero), ([zero, one], one), ([gamma, zero], -delta)]
    for x_row, x_rhs in x_options:
        for y_row, y_rhs in y_options:
            yield FaceSystem([x_row, y_row], [x_rhs, y_rhs])


def bimatrix_rest_points(payoff_data):
    """Rest points of a 2Pop2S game as (x, y), the first-strategy probabilities."""
    vertices = [[Fraction(x), Fraction(y)] for x in (0, 1) for y in (0, 1)]
    return enumerate_rest_points(bimatrix_faces(payoff_data), vertices)
