""":mod:`pvat.estimators` implements the value estimators."""

from abc import ABC, abstractmethod
from collections.abc import Container, Iterable, Iterator
from dataclasses import replace
from functools import partial, singledispatchmethod
from itertools import product, starmap
from typing import Any, Generic, TypeVar

from pvat.utilities import (
    ActionsFunction,
    LinearValueFunction,
    MadeFunction,
    MappedHistoriesFunction,
    MappedTerminalHistoriesFunction,
    PlayerFunction,
    PrefixesFunction,
    ProbabilityFunction,
    TerminalProbabilityFunction,
    TerminalValueFunction,
    ValueFunction,
    weighted_average_map,
)

_H = TypeVar('_H')
_A = TypeVar('_A')
_P = TypeVar('_P')


class ValueEstimator(ABC):
    """An abstract base class for value estimators.

    When invoked with a terminal history, an unbiased value estimate of
    the terminal history is calculated.
    """

    pass


class ImportanceSampling(ValueEstimator, Generic[_H]):
    """A class for value estimators with importance sampling.

    When invoked with a terminal history, an unbiased value estimate of
    the terminal history, using the technique of imaginary sampling over
    imaginary observations (Bowling et al. 2008), is calculated.

    :param terminal_value: The terminal value function.
    :param mapped_terminal_histories: The mapped terminal histories
                                      function.
    :param terminal_probability: The terminal probability function.
    """

    def __init__(
            self,
            terminal_value: TerminalValueFunction[_H],
            mapped_terminal_histories: MappedTerminalHistoriesFunction[_H],
            terminal_probability: TerminalProbabilityFunction[_H],
    ):
        self.terminal_value = terminal_value
        """The terminal value function."""
        self.mapped_terminal_histories = mapped_terminal_histories
        """The mapped terminal histories function."""
        self.terminal_probability = terminal_probability
        """The terminal probability function."""

    def __call__(self, terminal_history: _H) -> Any:
        return weighted_average_map(
            self.terminal_probability,
            self.terminal_value,
            self.mapped_terminal_histories(terminal_history),
        )


class AdvantageSum(ValueEstimator, Generic[_H, _A], ABC):
    """An abstract base class for value estimators with advantage sum.

    When invoked with a value function and a terminal history, an
    unbiased value estimate of the terminal history, using the technique
    of advantage sum (Zinkevich et al. 2006), is calculated.

    :param terminal_value: The terminal value function.
    :param actions: The actions function.
    :param probability: The probability function.
    :param made: The made function.
    """

    def __init__(
            self,
            terminal_value: TerminalValueFunction[_H],
            actions: ActionsFunction[_H, _A],
            probability: ProbabilityFunction[_H],
            made: MadeFunction[_H, _A],
    ) -> None:
        self.terminal_value = terminal_value
        """The terminal value function."""
        self.actions = actions
        """The actions function."""
        self.probability = probability
        """The probability function."""
        self.made = made
        """The made function."""

    def __call__(self, value: ValueFunction[_H], terminal_history: _H) -> Any:
        return (
            self.get_base_term(terminal_history)
            + self.get_correction_term_sum(value, terminal_history)
        )

    def get_base_term(self, terminal_history: _H) -> Any:
        """Calculate the base term.

        :param terminal: The terminal history.
        :return: The base term.
        """
        return self.terminal_value(terminal_history)

    @singledispatchmethod
    def get_correction_term_sum(
            self,
            value: ValueFunction[_H],
            terminal_history: _H,
    ) -> Any:
        """Calculate the sum of the correction terms.

        :param value: The value estimator used.
        :param terminal: The terminal history.
        :return: The sum of the correction terms.
        """
        correction_term_sum = 0

        for history, action in self.get_corrected_prefixes(terminal_history):
            correction_term_sum += (
                weighted_average_map(
                    self.probability,
                    value,
                    map(partial(self.made, history), self.actions(history)),
                )
                - value(self.made(history, action))
            )

        return correction_term_sum

    @get_correction_term_sum.register(LinearValueFunction)
    def _(
            self,
            value: LinearValueFunction[_H],
            terminal_history: _H,
    ) -> Any:
        value = replace(
            value,
            feature_extractor=partial(
                self.get_correction_term_sum,
                value.feature_extractor,
            ),
        )

        return value(terminal_history)

    @abstractmethod
    def get_corrected_prefixes(
            self,
            terminal_history: _H,
    ) -> Iterable[tuple[_H, _A]]:
        """Iterate through the prefixes from which correction terms are
        to be generated.

        :param terminal_history: The terminal history to iterate
                                 through.
        :return: An iterable of prefixes pending correction.
        """
        pass


class DIVAT(AdvantageSum[_H, _A], Generic[_H, _A, _P]):
    """A class for value estimators with DIVAT.

    When invoked with a value function and a terminal history, an
    unbiased value estimate of the terminal history, using the technique
    of DIVAT (Billings and Kan 2006, Zinkevich et al. 2006), is
    calculated.

    :param terminal_value: The terminal value function.
    :param actions: The actions function.
    :param probability: The probability function.
    :param made: The made function.
    :param prefixes: The prefixes function.
    :param player: The player function.
    :param corrected_players: The corrected players function.
    """

    def __init__(
            self,
            terminal_value: TerminalValueFunction[_H],
            actions: ActionsFunction[_H, _A],
            probability: ProbabilityFunction[_H],
            made: MadeFunction[_H, _A],
            prefixes: PrefixesFunction[_H, _A],
            player: PlayerFunction[_H, _P],
            corrected_players: Container[_P],
    ):
        super().__init__(terminal_value, actions, probability, made)

        self.prefixes = prefixes
        """The prefixes function."""
        self.player = player
        """The player function."""
        self.corrected_players = corrected_players
        """The corrected players function."""

    def get_corrected_prefixes(
            self,
            terminal_history: _H,
    ) -> Iterator[tuple[_H, _A]]:
        """Iterate through the prefixes whose player is among the
        corrected players.

        The corrected players have a known action distributions using
        which the correction term can be generated at each corrected
        prefixes.

        :param terminal_history: The terminal history to iterate
                                 through.
        :return: An iterable of prefixes whose player is in corrected
                 players.
        """
        for history, action in self.prefixes(terminal_history):
            if self.player(history) in self.corrected_players:
                yield history, action


class MIVAT(AdvantageSum[_H, _A], Generic[_H, _A, _P]):
    """A class for value estimators with MIVAT.

    When invoked with a value function and a terminal history, an
    unbiased value estimate of the terminal history, using the technique
    of MIVAT (White and Bowling 2009), is calculated.

    :param terminal_value: The terminal value function.
    :param actions: The actions function.
    :param probability: The probability function.
    :param made: The made function.
    :param prefixes: The prefixes function.
    :param player: The player function.
    :param chance: The chance (or nature).
    """

    def __init__(
            self,
            terminal_value: TerminalValueFunction[_H],
            actions: ActionsFunction[_H, _A],
            probability: ProbabilityFunction[_H],
            made: MadeFunction[_H, _A],
            prefixes: PrefixesFunction[_H, _A],
            player: PlayerFunction[_H, _P],
            chance: _P,
    ):
        super().__init__(terminal_value, actions, probability, made)

        self.prefixes = prefixes
        """The prefixes function."""
        self.player = player
        """The player function."""
        self.chance = chance
        """The chance (or nature)."""

    def get_corrected_prefixes(
            self,
            terminal_history: _H,
    ) -> Iterator[tuple[_H, _A]]:
        """Iterate through the prefixes whose actor is the chance (or
        nature).

        The chance has a known chance action (or event) distributions
        using which the correction term can be generated at each
        corrected prefixes.

        :param terminal_history: The terminal history to iterate
                                 through.
        :return: An iterable of prefixes whose player is the chance.
        """
        for history, action in self.prefixes(terminal_history):
            if self.player(history) == self.chance:
                yield history, action


class AIVAT(ValueEstimator, Generic[_H, _A, _P]):
    """A class for value estimators with AIVAT.

    When invoked with a value function and a terminal history, an
    unbiased value estimate of the terminal history, using the technique
    of AIVAT (Burch et al. 2018), is calculated.

    :param terminal_value: The terminal value function.
    :param mapped_terminal_histories: The mapped terminal histories
                                      function.
    :param terminal_probability: The terminal probability function.
    :param actions: The actions function.
    :param probability: The probability function.
    :param made: The made function.
    :param prefixes: The prefixes function.
    :param player: The player function.
    :param corrected_players: The corrected players function.
    :param mapped_histories: The mapped histories function.
    """

    def __init__(
            self,
            terminal_value: TerminalValueFunction[_H],
            mapped_terminal_histories: MappedTerminalHistoriesFunction[_H],
            terminal_probability: TerminalProbabilityFunction[_H],
            actions: ActionsFunction[_H, _A],
            probability: ProbabilityFunction[_H],
            made: MadeFunction[_H, _A],
            prefixes: PrefixesFunction[_H, _A],
            player: PlayerFunction[_H, _P],
            corrected_players: Container[_P],
            mapped_histories: MappedHistoriesFunction[_H],
    ):
        self.importance_sampling = ImportanceSampling(
            terminal_value,
            mapped_terminal_histories,
            terminal_probability,
        )
        """The importance sampling value estimator."""
        self.divat = DIVAT(
            terminal_value,
            actions,
            probability,
            made,
            prefixes,
            player,
            corrected_players,
        )
        """The DIVAT value estimator."""
        self.mapped_histories = mapped_histories
        """The mapped histories function."""

    def __call__(self, value: ValueFunction[_H], terminal_history: _H) -> Any:
        return (
            self.get_base_term(terminal_history)
            + self.get_correction_term_sum(value, terminal_history)
        )

    def get_base_term(self, terminal_history: _H) -> Any:
        """Calculate the base term.

        :param terminal: The terminal history.
        :return: The base term.
        """
        return self.importance_sampling(terminal_history)

    @singledispatchmethod
    def get_correction_term_sum(
            self,
            value: ValueFunction[_H],
            terminal_history: _H,
    ) -> Any:
        """Calculate the sum of the correction terms.

        :param value: The value estimator used.
        :param terminal: The terminal history.
        :return: The sum of the correction terms.
        """
        correction_term_sum = 0

        for history, action in self.divat.get_corrected_prefixes(
                terminal_history,
        ):
            correction_term_sum += (
                weighted_average_map(
                    self.divat.probability,
                    value,
                    starmap(
                        self.divat.made,
                        product(
                            self.mapped_histories(history),
                            self.divat.actions(history),
                        ),
                    ),
                )
                - weighted_average_map(
                    self.divat.probability,
                    value,
                    map(
                        lambda history: self.divat.made(history, action),
                        self.mapped_histories(history),
                    ),
                )
            )

        return correction_term_sum

    @get_correction_term_sum.register(LinearValueFunction)
    def _(
            self,
            value: LinearValueFunction[_H],
            terminal_history: _H,
    ) -> Any:
        value = replace(
            value,
            feature_extractor=partial(
                self.get_correction_term_sum,
                value.feature_extractor,
            ),
        )

        return value(terminal_history)
