"""Core abstractions for representing normal-form games."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

PayoffMatrix = np.ndarray
PayoffCollection = Tuple[PayoffMatrix, ...]
RawPayoffData = Union[
    PayoffMatrix,
    Sequence[Sequence[float]],
    Sequence[PayoffMatrix],
]

__all__ = [
    "Game",
    "PayoffMatrix",
    "PayoffCollection",
    "RawPayoffData",
    "format_game_description",
    "game_names",
    "infer_game_class",
]


class Game:
    """Normal-form game container used by pyNamo.

    The class stores payoff data, strategy labels, player labels, and basic
    metadata. It does not compute dynamics by itself; plotting and analysis
    modules consume Game objects.
    """

    def __init__(
        self,
        name: str,
        payoffs: RawPayoffData,
        *,
        strategy_labels: Optional[Sequence[str]] = None,
        player_strategy_labels: Optional[Sequence[Sequence[str]]] = None,
        player_labels: Optional[Sequence[str]] = None,
        description: str = "",
        reference: str = "",
        reference_note: str = "",
        parameters: str = "",
        illustrates: str = "",
        symmetric: Optional[bool] = None,
    ) -> None:
        """Create a normal-form game supported by pyNamo.

        Parameters
        ----------
        name : str
            Human-readable name of the game. Used in plot titles and example
            displays.
        payoffs : array-like or tuple of array-like
            Payoff data defining the game.

            For symmetric 2-player games, pass one square payoff matrix. Entry
            `payoffs[i, j]` is the payoff to a player using strategy i
            against an opponent using strategy j.

            For asymmetric 2-player / 2-strategy games, pass a tuple
            `(payoff_player_1, payoff_player_2)`, where both matrices have
            shape `(2, 2)`. Each matrix uses the focal-player convention: its
            rows are the strategies of the player receiving that matrix's
            payoffs, and its columns are the opponent's strategies. Thus
            `payoff_player_1[i, j]` is player 1's payoff when player 1 uses
            strategy i and player 2 uses strategy j, while
            `payoff_player_2[i, j]` is player 2's payoff when player 2 uses
            strategy i and player 1 uses strategy j.

            For asymmetric 3-player / 2-strategy games, pass a tuple
            `(payoff_player_1, payoff_player_2, payoff_player_3)`, where each
            payoff tensor has shape `(2, 2, 2)`. Entry `[i, j, k]` is evaluated
            at player 1 strategy i, player 2 strategy j, and player 3 strategy
            k.
        strategy_labels : sequence of str, optional
            Shared strategy labels. For symmetric games, these label the rows
            and columns of the payoff matrix. For asymmetric games, these are
            used as a fallback when all players have the same strategy labels.
        player_strategy_labels : sequence of sequence of str, optional
            Player-specific strategy labels. Use this for asymmetric games when
            players have different strategy names. For example:
            `[["Fight", "Flee"], ["Aggressive", "Cautious"]]`.
        player_labels : sequence of str, optional
            Labels for players or populations. Used in axis labels, payoff
            displays, and analysis output where relevant. For example:
            `["Prey", "Predator"]`.
        description : str, default=""
            Optional longer description of the game. Used by example catalogues
            and documentation-facing interfaces.
        reference : str, default=""
            Bibliographic source or conventional reference for catalogue
            examples. Empty for user-defined games unless supplied.
        reference_note : str, default=""
            Short clarification about the reference, for example whether the
            implemented payoff matrix is an exact source matrix, a parameter
            choice from a family, or a pedagogical variant.
        parameters : str, default=""
            Human-readable description of parameter values used to construct a
            catalogue example.
        illustrates : str, default=""
            Short statement of the mathematical or pedagogical point illustrated
            by the game.
        symmetric : bool or None, optional
            Whether the game is symmetric. If None, pyNamo infers symmetry from
            the payoff representation: one matrix means symmetric, a tuple of
            matrices or tensors means asymmetric. If supplied, the value must
            be consistent with the payoff representation.

        Attributes
        ----------
        name : str
            Human-readable game name.
        payoff_data : numpy.ndarray or tuple of numpy.ndarray
            Normalized payoff representation used internally by pyNamo.
        strategy_labels : list of str
            Shared strategy labels.
        player_strategy_labels : list of list of str
            Strategy labels for each player or population.
        player_labels : list of str
            Player or population labels.
        symmetric : bool
            Whether the game is treated as symmetric.
        game_class : str
            Supported pyNamo game-class identifier inferred from the payoff
            data. Currently one of "1Pop2S", "2Pop2S", "1Pop3S", "1Pop4S", "3Pop2S", or
            "unsupported".

        Notes
        -----
        pyNamo currently supports only games whose payoff data identify one of
        the implemented classes with directly visualizable state spaces:

        - "1Pop2S": symmetric 2-player / 2-strategy games in one population
        - "2Pop2S": asymmetric 2-player / 2-strategy games
        - "1Pop3S": symmetric 2-player / 3-strategy games
        - "1Pop4S": symmetric 2-player / 4-strategy games
        - "3Pop2S": asymmetric 3-player / 2-strategy games

        For equilibrium tables and trajectories, strategy order matters. In
        symmetric games, probability vectors follow `strategy_labels`. In
        2-strategy asymmetric games, reduced coordinates are probabilities of
        the first listed strategy for each player.
        """
        self.name = name
        self.description = description
        self.reference = reference
        self.reference_note = reference_note
        self.parameters = parameters
        self.illustrates = illustrates
        self.strategy_labels: List[str] = list(strategy_labels or [])
        self.player_strategy_labels: List[List[str]] = [
            list(labels) for labels in (player_strategy_labels or [])
        ]
        self.player_labels: List[str] = list(player_labels or [])
        self._payoff = self._normalize_payoff(payoffs)

        inferred_symmetry = not isinstance(self._payoff, tuple)
        if symmetric is None:
            self.symmetric = inferred_symmetry
        else:
            if symmetric != inferred_symmetry:
                raise ValueError(
                    f"Incompatible symmetry flag for game '{name}'. "
                    f"Expected {'symmetric' if inferred_symmetry else 'asymmetric'} matrices."
                )
            self.symmetric = symmetric

        self._validate_dimensions()
        self._ensure_label_defaults()
        self._ensure_player_label_defaults()

    @property
    def payoff_data(self) -> Union[PayoffMatrix, PayoffCollection]:
        """Return the underlying payoff representation."""
        return self._payoff

    @property
    def game_class(self) -> str:
        """Return the currently supported pyNamo game class identifier."""
        return infer_game_class(self._payoff)

    def describe(self) -> None:
        """Print a readable summary of the game and its metadata.

        The method is meant for quick inspection in notebooks and interactive
        sessions. Bibliographic metadata are optional, so empty fields are
        skipped.

        Examples
        --------
        >>> import pynamo_egt as pn
        >>> pn.examples.games.good_rps.describe()
        """
        print(format_game_description(self))

    def payoff_for_player(self, player: int = 0) -> PayoffMatrix:
        """Return the payoff matrix relevant for the requested player."""
        if self.symmetric:
            return self._payoff  # type: ignore[return-value]
        try:
            return self._payoff[player]  # type: ignore[index]
        except IndexError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Player index {player} out of range for game '{self.name}'."
            ) from exc

    def num_strategies(self) -> int:
        """Number of strategies available to each player."""
        if self.symmetric:
            return self._payoff.shape[0]  # type: ignore[return-value]
        return self._payoff[0].shape[0]  # type: ignore[index]

    def num_players(self) -> int:
        """Number of populations/players represented in the game."""
        if self.symmetric:
            return 1
        return len(self._payoff)  # type: ignore[arg-type]

    def strategy_labels_for_player(self, player: int) -> List[str]:
        """Return strategy labels for a specific population/player."""
        try:
            return self.player_strategy_labels[player]
        except IndexError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Player index {player} out of range for game '{self.name}'."
            ) from exc

    def expected_payoffs(self, mixed_strategies: Sequence[Sequence[float]]) -> np.ndarray:
        """Expected payoff to each pure strategy under the supplied mixed strategies."""
        strategy_count = self.num_strategies()

        if self.symmetric:
            if len(mixed_strategies) != 1:
                raise ValueError(
                    "Symmetric games expect a single population mixed strategy."
                )
            sigma = self._as_probability_vector(mixed_strategies[0], strategy_count)
            return self._payoff @ sigma  # type: ignore[return-value]

        players = len(self._payoff)  # type: ignore[arg-type]
        if len(mixed_strategies) != players:
            raise ValueError(
                f"Asymmetric games expect {players} mixed strategies (one per population)."
            )

        vectors = [
            self._as_probability_vector(sigma, strategy_count) for sigma in mixed_strategies
        ]

        payoffs = np.empty((players, strategy_count), dtype=float)
        for idx, matrix in enumerate(self._payoff):  # type: ignore[iterable]
            result = matrix
            for axis in reversed(range(players)):
                if axis == idx:
                    continue
                result = np.tensordot(result, vectors[axis], axes=([axis], [0]))
            payoffs[idx] = result
        return payoffs

    def _normalize_payoff(
        self, payoffs: RawPayoffData
    ) -> Union[PayoffMatrix, PayoffCollection]:
        if isinstance(payoffs, tuple):
            matrices = tuple(self._to_numpy(matrix) for matrix in payoffs)
            return matrices  # type: ignore[return-value]
        if isinstance(payoffs, list) and payoffs and isinstance(
            payoffs[0], np.ndarray
        ):
            matrices_tuple = tuple(self._to_numpy(matrix) for matrix in payoffs)  # type: ignore[arg-type]
            if len(matrices_tuple) == 1:
                return matrices_tuple[0]
            return matrices_tuple  # type: ignore[return-value]
        return self._to_numpy(payoffs)  # type: ignore[arg-type]

    def _to_numpy(self, matrix: Union[PayoffMatrix, Sequence[Sequence[float]]]) -> PayoffMatrix:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 2:
            rows, cols = arr.shape
            if rows != cols:
                raise ValueError("Payoff matrices must be square.")
        elif arr.ndim < 2:
            raise ValueError("Payoff arrays must have at least two dimensions.")
        return arr

    def _validate_dimensions(self) -> None:
        if self.symmetric:
            if self._payoff.ndim != 2:  # type: ignore[union-attr]
                raise ValueError(
                    "A single payoff array describes a symmetric game and must be a "
                    f"square matrix; got an array of shape {self._payoff.shape}. "  # type: ignore[union-attr]
                    "For an asymmetric game, pass a tuple with one payoff array per player."
                )
            if self._payoff.shape[0] < 2:  # type: ignore[index]
                raise ValueError("Symmetric games must have at least two strategies.")
        else:
            shapes = [matrix.shape for matrix in self._payoff]  # type: ignore[iterable]
            if len(set(shapes)) != 1:
                listed = ", ".join(str(shape) for shape in dict.fromkeys(shapes))
                raise ValueError(
                    f"All payoff arrays must have the same shape; got shapes {listed}."
                )
            players = len(shapes)
            rank = len(shapes[0])
            if rank != players:
                raise ValueError(
                    f"Got {players} payoff arrays of rank {rank}, but an asymmetric game "
                    f"with {players} players needs one payoff array of rank {players} "
                    "per player (one axis per player's strategy)."
                )

    def _ensure_label_defaults(self) -> None:
        if self.strategy_labels:
            return

        count = self.num_strategies()
        default_labels = [f"S{i+1}" for i in range(count)]
        self.strategy_labels = default_labels

    def _ensure_player_label_defaults(self) -> None:
        players = self.num_players()
        strategy_count = self.num_strategies()

        if self.player_strategy_labels:
            if len(self.player_strategy_labels) != players:
                raise ValueError(
                    "player_strategy_labels must provide one label list per player."
                )
            if any(len(labels) != strategy_count for labels in self.player_strategy_labels):
                raise ValueError(
                    "Each player_strategy_labels entry must match the number of strategies."
                )
        else:
            self.player_strategy_labels = [
                list(self.strategy_labels) for _ in range(players)
            ]

        if self.player_labels:
            if len(self.player_labels) != players:
                raise ValueError("player_labels must provide one label per player.")
        else:
            self.player_labels = (
                ["Population"] if self.symmetric else [f"Population {i+1}" for i in range(players)]
            )

    def _as_probability_vector(
        self, strategy: Sequence[float], expected_size: int
    ) -> np.ndarray:
        vector = np.asarray(strategy, dtype=float)
        if vector.ndim != 1:
            raise ValueError("Mixed strategies must be one-dimensional vectors.")
        if vector.size != expected_size:
            raise ValueError(
                f"Mixed strategy length {vector.size} incompatible with {expected_size} strategies."
            )
        if np.any(vector < -1e-12):
            raise ValueError("Mixed strategies cannot have negative probabilities.")

        total = vector.sum()
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError("Mixed strategies must sum to 1.")
        return vector


def game_names(games: Iterable[Tuple[int, Game]]) -> dict:
    """Return a simple id->name mapping derived from a catalogue of games."""
    return {idx: game.name for idx, game in games}


def format_game_description(game: Game) -> str:
    """Return a readable plain-text summary of a game and its metadata."""
    lines = [
        game.name,
        "-" * len(game.name),
        f"Game class: {game.game_class}",
        f"Symmetric: {game.symmetric}",
    ]

    optional_sections = [
        ("Description", game.description),
        ("Illustrates", game.illustrates),
        ("Parameters", game.parameters),
        ("Reference", game.reference),
        ("Reference note", game.reference_note),
    ]
    for label, value in optional_sections:
        if value:
            lines.extend(["", f"{label}:", value])

    lines.extend(
        [
            "",
            "Strategy labels:",
            str(game.strategy_labels),
            "",
            "Player strategy labels:",
            str(game.player_strategy_labels),
            "",
            "Player labels:",
            str(game.player_labels),
            "",
            "Payoff data:",
            str(game.payoff_data),
        ]
    )

    return "\n".join(lines)


def infer_game_class(game) -> str:
    """Infer the supported pyNamo game class from a Game or payoff data."""
    payoff_data = getattr(game, "payoff_data", game)

    if isinstance(payoff_data, np.ndarray):
        if payoff_data.shape == (2, 2):
            return "1Pop2S"
        if payoff_data.shape == (3, 3):
            return "1Pop3S"
        if payoff_data.shape == (4, 4):
            return "1Pop4S"
        return "unsupported"

    if isinstance(payoff_data, (tuple, list)) and payoff_data:
        first = payoff_data[0]
        if first.shape == (2, 2):
            return "2Pop2S"
        if first.shape == (2, 2, 2):
            return "3Pop2S"

    return "unsupported"
