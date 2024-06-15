""":mod:`pvat.estimators` implements the value estimators.

The value estimators implemented in PVAT are as follows:

- Importance Sampling (Bowling et al. 2008)
- Advantage Sum (Zinkevich et al. 2006)

  - DIVAT: Ignorant Value Assessment Tool (Billings and Kan 2006,
    Zinkevich et al. 2006)
  - MIVAT: Informed Value Assessment Tool (White and Bowling 2009)

- AIVAT: Action-Informed Value Assessment Tool (Burch et al. 2018)
"""

from abc import ABC, abstractmethod
from collections.abc import Container, Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
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


@dataclass
class ValueEstimator(ABC):
    """An abstract base class for value estimators."""

    pass


@dataclass
class ImportanceSampling(ValueEstimator, Generic[_H]):
    """A class for value estimators with importance sampling.

    When invoked with a terminal history, an unbiased value estimate of
    the terminal history, using the technique of imaginary sampling over
    imaginary observations (Bowling et al. 2008), is calculated.

    :param terminal_value_function: The terminal value function.
    :param mapped_terminal_histories_function: The mapped terminal
                                               histories function.
    :param terminal_probability_function: The terminal probability
                                          function.
    """

    terminal_value_function: TerminalValueFunction[_H]
    """The terminal value function."""
    mapped_terminal_histories_function: MappedTerminalHistoriesFunction[_H]
    """The mapped terminal histories function."""
    terminal_probability_function: TerminalProbabilityFunction[_H]
    """The terminal probability function."""

    def __call__(self, terminal_history: _H) -> Any:
        return weighted_average_map(
            self.terminal_probability_function,
            self.terminal_value_function,
            self.mapped_terminal_histories_function(terminal_history),
        )


@dataclass
class ValueFunctionEstimator(ValueEstimator, Generic[_H], ABC):
    """An abstract base class for value estimators that uses value
    functions.

    When invoked with a value function and a terminal history, an
    unbiased value estimate of the terminal history is calculated.
    """

    def __call__(
            self,
            value_function: ValueFunction[_H],
            terminal_history: _H,
    ) -> Any:
        return (
            self.get_base_term(terminal_history)
            + self.get_correction_term_sum(value_function, terminal_history)
        )

    @abstractmethod
    def get_base_term(self, terminal_history: _H) -> Any:
        """Calculate the base term.

        :param terminal: The terminal history.
        :return: The base term.
        """
        pass

    @abstractmethod
    def get_correction_term_sum(
            self,
            value_function: ValueFunction[_H],
            terminal_history: _H,
    ) -> Any:
        """Calculate the sum of the correction terms.

        :param value_function: The value estimator used.
        :param terminal: The terminal history.
        :return: The sum of the correction terms.
        """
        pass


@dataclass
class AdvantageSum(ValueFunctionEstimator[_H], Generic[_H, _A], ABC):
    """An abstract base class for value estimators with advantage sum.

    When invoked with a value function and a terminal history, an
    unbiased value estimate of the terminal history, using the technique
    of advantage sum (Zinkevich et al. 2006), is calculated.

    :param terminal_value_function: The terminal value function.
    :param actions_function: The actions_function function.
    :param probability_function: The probability function.
    :param made_function: The made_function function.
    """

    terminal_value_function: TerminalValueFunction[_H]
    """The terminal value function."""
    actions_function: ActionsFunction[_H, _A]
    """The actions_function function."""
    probability_function: ProbabilityFunction[_H]
    """The probability function."""
    made_function: MadeFunction[_H, _A]
    """The made_function function."""

    def get_base_term(self, terminal_history: _H) -> Any:
        return self.terminal_value_function(terminal_history)

    @LinearValueFunction.optimize_method
    def get_correction_term_sum(
            self,
            value_function: ValueFunction[_H],
            terminal_history: _H,
    ) -> Any:
        correction_term_sum = 0

        for history, action in self.get_corrected_prefixes(terminal_history):
            correction_term_sum += (
                weighted_average_map(
                    self.probability_function,
                    value_function,
                    map(
                        partial(self.made_function, history),
                        self.actions_function(history),
                    ),
                )
                - value_function(self.made_function(history, action))
            )

        return correction_term_sum

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


@dataclass
class DIVAT(AdvantageSum[_H, _A], Generic[_H, _A, _P]):
    """A class for value estimators with DIVAT.

    When invoked with a value function and a terminal history, an
    unbiased value estimate of the terminal history, using the technique
    of DIVAT (Billings and Kan 2006, Zinkevich et al. 2006), is
    calculated.

    :param terminal_value_function: The terminal value function.
    :param actions_function: The actions_function function.
    :param probability_function: The probability function.
    :param made_function: The made_function function.
    :param prefixes_function: The prefixes function.
    :param player_function: The player function.
    :param corrected_players: The corrected players function.
    """

    prefixes_function: PrefixesFunction[_H, _A]
    """The prefixes function."""
    player_function: PlayerFunction[_H, _P]
    """The player function."""
    corrected_players: Container[_P]
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
        for history, action in self.prefixes_function(terminal_history):
            if self.player_function(history) in self.corrected_players:
                yield history, action


@dataclass
class MIVAT(AdvantageSum[_H, _A], Generic[_H, _A, _P]):
    """A class for value estimators with MIVAT.

    When invoked with a value function and a terminal history, an
    unbiased value estimate of the terminal history, using the technique
    of MIVAT (White and Bowling 2009), is calculated.

    :param terminal_value_function: The terminal value function.
    :param actions_function: The actions_function function.
    :param probability_function: The probability function.
    :param made_function: The made_function function.
    :param prefixes_function: The prefixes function.
    :param player_function: The player function.
    :param chance: The chance (or nature).
    """

    prefixes_function: PrefixesFunction[_H, _A]
    """The prefixes function."""
    player_function: PlayerFunction[_H, _P]
    """The player function."""
    chance: _P
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
        for history, action in self.prefixes_function(terminal_history):
            if self.player_function(history) == self.chance:
                yield history, action


@dataclass
class AIVAT(ValueFunctionEstimator[_H], Generic[_H, _A, _P]):
    """A class for value estimators with AIVAT.

    When invoked with a value function and a terminal history, an
    unbiased value estimate of the terminal history, using the technique
    of AIVAT (Burch et al. 2018), is calculated.

    :param terminal_value_function: The terminal value function.
    :param mapped_terminal_histories_function: The mapped terminal
                                               histories function.
    :param terminal_probability_function: The terminal probability
                                          function.
    :param actions_function: The actions_function function.
    :param probability_function: The probability function.
    :param made_function: The made_function function.
    :param prefixes_function: The prefixes function.
    :param player_function: The player function.
    :param corrected_players: The corrected players function.
    :param mapped_histories_function: The mapped histories function.
    """

    terminal_value_function: TerminalValueFunction[_H]
    """The terminal value function."""
    mapped_terminal_histories_function: MappedTerminalHistoriesFunction[_H]
    """The mapped terminal histories function."""
    terminal_probability_function: TerminalProbabilityFunction[_H]
    """The terminal probability function."""
    actions_function: ActionsFunction[_H, _A]
    """The actions_function function."""
    probability_function: ProbabilityFunction[_H]
    """The probability function."""
    made_function: MadeFunction[_H, _A]
    """The made_function function."""
    prefixes_function: PrefixesFunction[_H, _A]
    """The prefixes function."""
    player_function: PlayerFunction[_H, _P]
    """The player function."""
    corrected_players: Container[_P]
    """The corrected players function."""
    mapped_histories_function: MappedHistoriesFunction[_H]
    """The mapped histories function."""
    _importance_sampling: ImportanceSampling[_H] = field(init=False)
    _divat: DIVAT[_H, _A, _P] = field(init=False)

    def __post_init__(self) -> None:
        self._importance_sampling = ImportanceSampling(
            self.terminal_value_function,
            self.mapped_terminal_histories_function,
            self.terminal_probability_function,
        )
        self._divat = DIVAT(
            self.terminal_value_function,
            self.actions_function,
            self.probability_function,
            self.made_function,
            self.prefixes_function,
            self.player_function,
            self.corrected_players,
        )

    def get_base_term(self, terminal_history: _H) -> Any:
        return self._importance_sampling(terminal_history)

    @LinearValueFunction.optimize_method
    def get_correction_term_sum(
            self,
            value_function: ValueFunction[_H],
            terminal_history: _H,
    ) -> Any:
        correction_term_sum = 0

        for history, action in self._divat.get_corrected_prefixes(
                terminal_history,
        ):
            correction_term_sum += (
                weighted_average_map(
                    self.probability_function,
                    value_function,
                    starmap(
                        self.made_function,
                        product(
                            self.mapped_histories_function(history),
                            self.actions_function(history),
                        ),
                    ),
                )
                - weighted_average_map(
                    self.probability_function,
                    value_function,
                    map(
                        lambda history: self.made_function(history, action),
                        self.mapped_histories_function(history),
                    ),
                )
            )

        return correction_term_sum
