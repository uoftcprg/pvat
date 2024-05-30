""":mod:`pvat.utilities` implements classes related to utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import starmap
from typing import Any, Protocol, TypeVar

from numpy.linalg import inv
import numpy as np

_H = TypeVar('_H')
_H_contra = TypeVar('_H_contra', contravariant=True)
_A_co = TypeVar('_A_co', covariant=True)
_A_contra = TypeVar('_A_contra', contravariant=True)
_P_co = TypeVar('_P_co', covariant=True)
_T = TypeVar('_T')


class TerminalValueFunction(Protocol[_H_contra]):
    """A protocol for a value function for terminal histories."""

    def __call__(self, terminal_history: _H_contra) -> Any: ...


class MappedTerminalHistoriesFunction(Protocol[_H]):
    """A protocol for a mapping function for terminal histories."""

    def __call__(self, terminal_history: _H) -> Iterable[_H]: ...


class TerminalProbabilityFunction(Protocol[_H_contra]):
    """A protocol for a probability function for terminal histories."""

    def __call__(self, history: _H_contra) -> Any: ...


class ActionsFunction(Protocol[_H_contra, _A_co]):
    """A protocol for an actions function for histories."""

    def __call__(self, history: _H_contra) -> Iterable[_A_co]: ...


class ProbabilityFunction(Protocol[_H_contra]):
    """A protocol for a probability function for histories."""

    def __call__(self, history: _H_contra) -> Any: ...


class MadeFunction(Protocol[_H, _A_contra]):
    """A protocol for a made function for history-actions."""

    def __call__(self, history: _H, action: _A_contra) -> _H: ...


class PrefixesFunction(Protocol[_H, _A_co]):
    """A protocol for a prefixes function for terminal histories."""

    def __call__(self, terminal_history: _H) -> Iterable[tuple[_H, _A_co]]: ...


class PlayerFunction(Protocol[_H_contra, _P_co]):
    """A protocol for a player function for histories."""

    def __call__(self, history: _H_contra) -> _P_co: ...


class MappedHistoriesFunction(Protocol[_H]):
    """A protocol for a mapping function for histories."""

    def __call__(self, terminal_history: _H) -> Iterable[_H]: ...


class ValueFunction(Protocol[_H_contra]):
    """A protocol for a value function for history-actions."""

    def __call__(self, history: _H_contra) -> Any: ...


@dataclass
class LinearValueFunction(ValueFunction[_H_contra]):
    """A protocol for a linear value function for history-actions.

    :param feature_extractor: The function that maps histories to a
                              vector of features.
    :param parameters: The parameters of the linear value function.
    """

    feature_extractor: Callable[[_H_contra], Any]
    """The function that maps histories to a vector of features."""
    parameters: Any
    """The parameters of the linear value function."""

    @classmethod
    def learn(
            cls,
            base_term: Callable[[_H_contra], Any],
            correction_term_sum: (
                Callable[[Callable[[_H_contra], Any], _H_contra], Any]
            ),
            feature_extractor: Callable[[_H_contra], Any],
            terminal_histories: Iterable[_H_contra],
            zero_sum: bool = False,
    ) -> LinearValueFunction[_H_contra]:
        """Solve the value estimator for the linear case.

        In two-player, zero-sum games, it is sufficient to learn just a
        single value function for one of the two players only (maybe via
        this method). The negation of the value estimator for the
        learned player is the variance minimizing estimator for the
        other player.

        For n-player, general-sum games, a value function should be
        learned for each player. However, for n-player, zero-sum games,
        this approach would not give a zero-sum guarantee. As such,
        for the case of linear value functions in n-player, zero-sum
        games, the zero-sum option must be activated.

        :param base_term: The base term function.
        :param correction_term_sum: The correction term function.
        :param feature_extractor: The feature extractor function.
        :param terminal_histories: The terminal histories to learn from.
        :param zero_sum: ``True`` to enable the zero-sum constraint, else
                         ``False``. This is only relevant when value is a
                         vector, not a scalar.
        :return: The solution to the value function for the linear case.
        """
        b = []
        A = []

        for terminal_history in terminal_histories:
            b.append(base_term(terminal_history))
            A.append(correction_term_sum(feature_extractor, terminal_history))

        b_bar = np.mean(b, 0)
        A_bar = np.mean(A, 0)

        A_t_A_t_T_mean = np.mean(tuple(starmap(np.outer, zip(A, A))), 0)
        A_bar_A_bar_T = np.outer(A_bar, A_bar)

        A_bar_b_bar_T = np.outer(A_bar, b_bar)
        A_t_b_t_T_mean = np.mean(tuple(starmap(np.outer, zip(A, b))), 0)

        parameters = (
            inv(A_t_A_t_T_mean - A_bar_A_bar_T)
            @ (A_bar_b_bar_T - A_t_b_t_T_mean)
        )

        if zero_sum:
            player_count = len(b_bar)
            parameters @= np.eye(player_count) - 1 / player_count

        return LinearValueFunction(feature_extractor, parameters)

    def __call__(self, terminal_history: _H_contra) -> Any:
        return self.parameters.T @ self.feature_extractor(terminal_history)


def weighted_average_map(
        weight: Callable[[_T], Any],
        value: Callable[[_T], Any],
        values: Iterable[_T],
) -> Any:
    """Perform weighted average where weights and values are mapped from
    an iterable.

    >>> from math import sqrt
    >>> from operator import neg
    >>> weighted_average_map(sqrt, neg, [1, 4, 9, 16, 25])
    -15.0

    :param weight: The weight function.
    :param value: The value function.
    :param values: The values to map.
    :return: The weighted average of the mapped values.
    """
    values = tuple(values)
    weights = tuple(map(weight, values))
    mapped_values = tuple(map(value, values))

    return np.average(mapped_values, axis=0, weights=weights)
