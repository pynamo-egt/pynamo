"""Exact replicator rest points by support (face) enumeration.

On each face of the state space, the replicator rest-point conditions are a
small linear system: every strategy in the support earns the same payoff.
Solving these systems in exact rationals finds every rest point (including
those outside the simplex, which callers filter) and detects continua
without a general-purpose symbolic solver, so analysis cannot hang.
"""

from __future__ import annotations

from fractions import Fraction
import math
from itertools import combinations, product
from typing import Iterable, Iterator, Literal, Optional, Protocol, Sequence, Union

import numpy as np

# Rational rest points are exact; irrational roots of a quadratic are floats.
Scalar = Union[Fraction, float]
Point = Sequence[Scalar]
Matrix = list[list[Fraction]]
SolveKind = Literal["unique", "continuum", "none"]
# a + b*u + c*w + d*u*w, and constant + slope*z.
Bilinear = tuple[Fraction, Fraction, Fraction, Fraction]
Linear = tuple[Fraction, Fraction]


class Face(Protocol):
    """Rest-point conditions on one face of the state space."""

    def solutions(self) -> tuple[list[Point], bool]:
        """Return ``(isolated_points, has_continuum)``."""

    def contains(self, point: Point) -> bool:
        """Whether ``point`` lies on a continuum of this face."""


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

    def solutions(self) -> tuple[list[Point], bool]:
        """Return ``(isolated_points, has_continuum)``."""
        kind, point = solve_linear(self.matrix, self.rhs)
        return ([point] if point is not None else []), kind == "continuum"

    def contains(self, point: Point) -> bool:
        """Whether ``point`` satisfies every equation of this face."""
        return all(
            _is_zero(sum(a * x for a, x in zip(row, point)) - b)
            for row, b in zip(self.matrix, self.rhs)
        )


def _is_zero(value: Scalar) -> bool:
    # Irrational rest points (roots of a quadratic) are carried as floats.
    return value == 0 if isinstance(value, Fraction) else abs(value) <= 1e-9


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
    faces: Iterable[Face], vertices: Sequence[Point]
) -> tuple[list[Point], bool]:
    """Collect rest points face by face, preserving the legacy output contract.

    Returns ``(points, degenerate)``. ``points`` lists the isolated rest points
    in lexicographic order, then any vertex not already listed, in the order
    given. Non-vertex points lying on a continuum are dropped; vertices are
    always listed. ``degenerate`` is True when some face carries a continuum.
    """
    isolated: list[Point] = []
    continua: list[Face] = []
    for face in faces:
        points, has_continuum = face.solutions()
        if has_continuum:
            continua.append(face)
        isolated.extend(point for point in points if point not in isolated)

    isolated = sorted(
        (point for point in isolated if not any(face.contains(point) for face in continua)),
        key=tuple,
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


def _indifference(payoff_matrix: np.ndarray) -> tuple[Fraction, Fraction]:
    """Coefficients (slope, intercept) of u(first) - u(second) in the opponent's
    probability of their first strategy, for a focal-player 2×2 matrix."""
    (a, b), (c, d) = [[to_fraction(value) for value in row] for row in payoff_matrix]
    return a - b - c + d, b - d


def bimatrix_faces(payoff_data: Sequence[np.ndarray]) -> Iterator[FaceSystem]:
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


def bimatrix_rest_points(payoff_data: Sequence[np.ndarray]) -> tuple[list[Point], bool]:
    """Rest points of a 2Pop2S game as (x, y), the first-strategy probabilities."""
    vertices = [[Fraction(x), Fraction(y)] for x in (0, 1) for y in (0, 1)]
    return enumerate_rest_points(bimatrix_faces(payoff_data), vertices)


def _cube_indifference(tensor: np.ndarray, player: int) -> Bilinear:
    """Coefficients (a, b, c, d) of u(first) - u(second) for ``player`` as
    a + b*u + c*w + d*u*w, where u and w are the other players' first-strategy
    probabilities in increasing player order."""
    T = [[[to_fraction(tensor[i][j][k]) for k in (0, 1)] for j in (0, 1)] for i in (0, 1)]

    def difference(o1: int, o2: int) -> Fraction:
        index = [o1, o2]
        index.insert(player, 0)
        first = T[index[0]][index[1]][index[2]]
        index[player] = 1
        return first - T[index[0]][index[1]][index[2]]

    c00, c01, c10, c11 = difference(0, 0), difference(0, 1), difference(1, 0), difference(1, 1)
    return c11, c01 - c11, c10 - c11, c00 - c01 - c10 + c11


def _linear_roots(constant: Fraction, slope: Fraction) -> Optional[list[Fraction]]:
    """Roots of constant + slope*z; None means identically zero."""
    if slope != 0:
        return [-constant / slope]
    return None if constant == 0 else []


def _quadratic_roots(q0: Fraction, q1: Fraction, q2: Fraction) -> Optional[Sequence[Scalar]]:
    """Real roots of q0 + q1*z + q2*z**2 (Fractions when rational, else floats);
    None means identically zero."""
    if q2 == 0:
        return _linear_roots(q0, q1)
    disc = q1 * q1 - 4 * q2 * q0
    if disc < 0:
        return []
    num, den = math.isqrt(disc.numerator), math.isqrt(disc.denominator)
    if num * num == disc.numerator and den * den == disc.denominator:
        root = Fraction(num, den)
        return sorted({(-q1 - root) / (2 * q2), (-q1 + root) / (2 * q2)})
    sqrt_disc = math.sqrt(disc)
    return sorted((float(-q1) - sign * sqrt_disc) / float(2 * q2) for sign in (1, -1))


def _at(poly: Linear, z: Scalar) -> Scalar:
    """Evaluate constant + slope*z."""
    return poly[0] + poly[1] * z


class MixedCubeFace:
    """Rest-point conditions where all three 3Pop2S players are indifferent.

    E1(y, z) = A1(z) + B1(z)*y, E2(x, z) = A2(z) + B2(z)*x and E3(x, y) are
    bilinear. For z with B1(z)*B2(z) != 0, E1 and E2 fix y and x, and E3
    becomes one quadratic Q(z). The finitely many other z are handled case by
    case. Zeros of this face that are not isolated form a continuum.
    """

    def __init__(self, e1: Bilinear, e2: Bilinear, e3: Bilinear) -> None:
        self.e1, self.e2, self.e3 = e1, e2, e3
        self._isolated, self._has_continuum = self._solve()

    def solutions(self) -> tuple[list[Point], bool]:
        """Return ``(isolated_points, has_continuum)``."""
        return list(self._isolated), self._has_continuum

    def contains(self, point: Point) -> bool:
        """Whether ``point`` is a non-isolated zero of this face."""
        return (
            self._has_continuum
            and all(_is_zero(value) for value in self._residuals(point))
            and point not in self._isolated
        )

    def _residuals(self, point: Point) -> tuple[Scalar, Scalar, Scalar]:
        x, y, z = point
        (a1, b1, c1, d1), (a2, b2, c2, d2), (a3, b3, c3, d3) = self.e1, self.e2, self.e3
        return (
            a1 + b1 * y + c1 * z + d1 * y * z,
            a2 + b2 * x + c2 * z + d2 * x * z,
            a3 + b3 * x + c3 * y + d3 * x * y,
        )

    def _solve(self) -> tuple[list[Point], bool]:
        (a1, b1, c1, d1), (a2, b2, c2, d2), (a3, b3, c3, d3) = self.e1, self.e2, self.e3
        A1, B1 = (a1, c1), (b1, d1)
        A2, B2 = (a2, c2), (b2, d2)
        points: list[Point] = []
        continuum = False

        b1_roots, b2_roots = _linear_roots(*B1), _linear_roots(*B2)
        if b1_roots is None or b2_roots is None:
            # A player's indifference does not involve their partner's
            # coordinate: E1 or E2 only constrains z.
            candidates: Optional[list[Fraction]] = None
            for A, B_roots in ((A1, b1_roots), (A2, b2_roots)):
                if B_roots is None:
                    roots = _linear_roots(*A)
                    if roots is None:
                        return [], True
                    candidates = roots if candidates is None else [z for z in candidates if z in roots]
            special: list[Fraction] = candidates or []
        else:
            q = [
                a3 * b1 * b2 - b3 * a2 * b1 - c3 * a1 * b2 + d3 * a1 * a2,
                a3 * (b1 * d2 + d1 * b2) - b3 * (a2 * d1 + c2 * b1)
                - c3 * (a1 * d2 + c1 * b2) + d3 * (a1 * c2 + c1 * a2),
                a3 * d1 * d2 - b3 * c2 * d1 - c3 * c1 * d2 + d3 * c1 * c2,
            ]
            generic = _quadratic_roots(*q)
            if generic is None:
                return [], True
            for z in generic:
                if _at(B1, z) != 0 and _at(B2, z) != 0:
                    points.append([-_at(A2, z) / _at(B2, z), -_at(A1, z) / _at(B1, z), z])
            special = sorted(set(b1_roots) | set(b2_roots))

        for z in special:
            found, line = self._at_special_z(z)
            points.extend(found)
            continuum = continuum or line
        return points, continuum

    def _at_special_z(self, z: Fraction) -> tuple[list[Point], bool]:
        """Solve E1 = E2 = E3 = 0 with z fixed where B1(z) or B2(z) vanishes."""
        (a1, b1, c1, d1), (a2, b2, c2, d2), (a3, b3, c3, d3) = self.e1, self.e2, self.e3
        A1, B1 = a1 + c1 * z, b1 + d1 * z
        A2, B2 = a2 + c2 * z, b2 + d2 * z
        if (B1 == 0 and A1 != 0) or (B2 == 0 and A2 != 0):
            return [], False
        if B1 == 0 and B2 == 0:
            # E3 alone constrains (x, y): a curve, a plane, or nothing.
            return [], any(value != 0 for value in (b3, c3, d3)) or a3 == 0
        if B1 == 0:
            x = -A2 / B2
            kind, solution = solve_linear([[c3 + d3 * x]], [-(a3 + b3 * x)])
            return ([[x, solution[0], z]] if solution else []), kind == "continuum"
        y = -A1 / B1
        kind, solution = solve_linear([[b3 + d3 * y]], [-(a3 + c3 * y)])
        return ([[solution[0], y, z]] if solution else []), kind == "continuum"


def cube_faces(payoff_tensors: Sequence[np.ndarray]) -> Iterator[Face]:
    """Yield the rest-point conditions on every face of a 3Pop2S cube."""
    zero, one = Fraction(0), Fraction(1)
    indifference = [_cube_indifference(tensor, player) for player, tensor in enumerate(payoff_tensors)]
    # Each coordinate is fixed at 0, fixed at 1, or free with its player indifferent.
    options: tuple[Optional[Fraction], ...]
    for options in product((zero, one, None), repeat=3):
        if all(option is None for option in options):
            yield MixedCubeFace(*indifference)
            continue
        matrix: Matrix = []
        rhs: list[Fraction] = []
        for player, option in enumerate(options):
            if option is not None:
                matrix.append([one if j == player else zero for j in range(3)])
                rhs.append(option)
                continue
            a, b, c, d = indifference[player]
            u, w = [j for j in range(3) if j != player]
            fixed_u, fixed_w = options[u], options[w]
            row = [zero] * 3
            if fixed_u is None:
                assert fixed_w is not None  # not every player is indifferent here
                row[u] = b + d * fixed_w
                rhs.append(-(a + c * fixed_w))
            elif fixed_w is None:
                row[w] = c + d * fixed_u
                rhs.append(-(a + b * fixed_u))
            else:
                rhs.append(-(a + b * fixed_u + c * fixed_w + d * fixed_u * fixed_w))
            matrix.append(row)
        yield FaceSystem(matrix, rhs)


def cube_rest_points(payoff_tensors: Sequence[np.ndarray]) -> tuple[list[Point], bool]:
    """Rest points of a 3Pop2S game as (x, y, z), the first-strategy probabilities."""
    vertices = [[Fraction(v) for v in vertex] for vertex in product((0, 1), repeat=3)]
    return enumerate_rest_points(cube_faces(payoff_tensors), vertices)
