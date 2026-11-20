"""State spaces: products of simplices, one per population."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .game import require_supported_game_class

__all__ = ["StateSpace"]

_POPULATIONS = {
    "1Pop2S": (2,),
    "1Pop3S": (3,),
    "1Pop4S": (4,),
    "2Pop2S": (2, 2),
    "3Pop2S": (2, 2, 2),
}
_KINDS = {
    (2,): "line",
    (3,): "triangle",
    (4,): "tetrahedron",
    (2, 2): "square",
    (2, 2, 2): "cube",
}


@dataclass(frozen=True)
class StateSpace:
    """Product of simplices, one per population.

    Reduced coordinates drop each population's last strategy, so a point has
    ``sum(n - 1)`` reduced coordinates; full coordinates list every strategy of
    every population, population by population.

    Parameters
    ----------
    populations : tuple of int
        Number of strategies of each population, e.g. ``(3,)`` for 1Pop3S.
    """

    populations: tuple[int, ...]

    @classmethod
    def for_game(cls, game) -> StateSpace:
        """State space of a supported pyNamo game."""
        return cls(_POPULATIONS[require_supported_game_class(game)])

    @property
    def dimension(self) -> int:
        """Number of reduced coordinates."""
        return sum(n - 1 for n in self.populations)

    @property
    def kind(self) -> str:
        """Name of the drawable shape, e.g. ``"triangle"``, or ``"product"``."""
        return _KINDS.get(self.populations, "product")

    def to_full(self, reduced: np.ndarray) -> np.ndarray:
        """Full coordinates, shape ``(..., sum(n))``, from reduced ones, shape ``(..., dimension)``."""
        reduced = np.asarray(reduced, dtype=float)
        parts, start = [], 0
        for n in self.populations:
            block = reduced[..., start:start + n - 1]
            parts += [block, 1.0 - block.sum(axis=-1, keepdims=True)]
            start += n - 1
        return np.concatenate(parts, axis=-1)

    def to_reduced(self, full: np.ndarray) -> np.ndarray:
        """Reduced coordinates from full ones (inverse of :meth:`to_full`)."""
        full = np.asarray(full, dtype=float)
        parts, start = [], 0
        for n in self.populations:
            parts.append(full[..., start:start + n - 1])
            start += n
        return np.concatenate(parts, axis=-1)

    def default_starts(self, count: int, random_state: int | None = None) -> list[list[float]]:
        """Reduced starting points spread over the space.

        Without ``random_state``, the centroids of a deterministic recursive
        partition into ``count`` cells; with it, seeded random states.
        """
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("trajectory_number must be a positive integer.")
        n = self.populations[0]
        simplex = len(self.populations) == 1 and n > 2
        box = all(m == 2 for m in self.populations)
        if not (simplex or box):
            raise NotImplementedError(f"No default starts for a {self.kind} yet.")
        if random_state is not None:
            rng = np.random.default_rng(random_state)
            if simplex:
                return rng.dirichlet(np.ones(n), size=count)[:, :-1].tolist()
            return rng.uniform(0.05, 0.95, size=(count, self.dimension)).tolist()
        if simplex:
            return _simplex_partition_centroids(n, count)[:, :-1].tolist()
        return _box_partition_centroids(self.dimension, count).tolist()

    def embed(self, reduced: np.ndarray) -> np.ndarray:
        """Plot coordinates of reduced states, matching ``phase_portrait``'s layout."""
        reduced = np.asarray(reduced, dtype=float)
        if self.kind == "triangle":
            # Strategy 1 at the top, strategy 2 bottom left, strategy 3 bottom right.
            x, y = reduced[..., 0], reduced[..., 1]
            return np.stack([-0.5 * x - y + 1, (np.sqrt(3) / 2) * x], axis=-1)
        raise NotImplementedError(f"No plot embedding for a {self.kind} yet.")


def _simplex_partition_centroids(strategy_count, count):
    """Return centroids of a deterministic recursive simplex partition."""
    cells = [np.eye(strategy_count)]

    def volume_measure(vertices):
        edges = vertices[1:] - vertices[0]
        return np.sqrt(max(np.linalg.det(edges @ edges.T), 0.0))

    while len(cells) < count:
        cell_index = max(range(len(cells)), key=lambda index: volume_measure(cells[index]))
        vertices = cells.pop(cell_index)
        pairs = [
            (i, j)
            for i in range(strategy_count)
            for j in range(i + 1, strategy_count)
        ]
        first, second = max(
            pairs,
            key=lambda pair: np.linalg.norm(vertices[pair[0]] - vertices[pair[1]]),
        )
        midpoint = (vertices[first] + vertices[second]) / 2
        child_1 = vertices.copy()
        child_2 = vertices.copy()
        child_1[first] = midpoint
        child_2[second] = midpoint
        cells.extend([child_1, child_2])

    return np.asarray([cell.mean(axis=0) for cell in cells])


def _box_partition_centroids(dimension, count):
    """Return centroids of a deterministic recursive unit-box partition."""
    boxes = [(np.zeros(dimension), np.ones(dimension))]
    while len(boxes) < count:
        box_index = max(
            range(len(boxes)),
            key=lambda index: np.prod(boxes[index][1] - boxes[index][0]),
        )
        lower, upper = boxes.pop(box_index)
        split_dimension = int(np.argmax(upper - lower))
        midpoint = (lower[split_dimension] + upper[split_dimension]) / 2
        upper_1 = upper.copy()
        upper_1[split_dimension] = midpoint
        lower_2 = lower.copy()
        lower_2[split_dimension] = midpoint
        boxes.extend([(lower, upper_1), (lower_2, upper)])

    return np.asarray([(lower + upper) / 2 for lower, upper in boxes])
