""":mod:`pvat.utilities` implements classes related to utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from functools import partial, singledispatchmethod
from typing import Any, Protocol, TypeVar

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
class LinearValueFunction(ValueFunction[_H]):
    """A protocol for a linear value function for history-actions.

    :param feature_extractor: The function that maps histories to a
                              vector of features.
    :param parameters: The parameters of the linear value function.
    """

    feature_extractor: Callable[[_H], Any]
    """The function that maps histories to a vector of features."""
    parameters: Any
    """The parameters of the linear value function."""

    @classmethod
    def optimize_method(
            cls,
            method: Callable[[Any, ValueFunction[_H], _H], Any],
    ) -> Callable[[Any, ValueFunction[_H], _H], Any]:
        optimized_method = singledispatchmethod(method)

        @optimized_method.register(LinearValueFunction)
        def _(
                self: Any,
                value_function: LinearValueFunction[_H],
                terminal_history: _H,
        ) -> Any:
            value_function = replace(
                value_function,
                feature_extractor=partial(
                    method,
                    self,
                    value_function.feature_extractor,
                ),
            )

            return value_function(terminal_history)

        return optimized_method  # type: ignore[return-value]

    @classmethod
    def learn(
            cls,
            base_term_function: Callable[[_H], Any],
            correction_term_sum_function: (
                Callable[[Callable[[_H], Any], _H], Any]
            ),
            feature_extractor: Callable[[_H], Any],
            terminal_histories: Iterable[_H],
            zero_sum: bool = False,
    ) -> LinearValueFunction[_H]:
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

        :param base_term_function: The base term function.
        :param correction_term_sum_function: The correction term
                                             function.
        :param feature_extractor: The feature extractor function.
        :param terminal_histories: The terminal histories to learn from.
        :param zero_sum: ``True`` to enable the zero-sum constraint,
                         else ``False``. This is only relevant when
                         value is a vector, not a scalar.
        :return: The solution to the value function for the linear case.
        """
        b_t = []
        A_t = []

        for terminal_history in terminal_histories:
            b_t.append(base_term_function(terminal_history))
            A_t.append(
                correction_term_sum_function(
                    feature_extractor,
                    terminal_history,
                ),
            )

        b_bar = np.mean(b_t, 0)
        A_bar = np.mean(A_t, 0)
        b = np.subtract(b_bar, b_t)
        A = np.subtract(A_t, A_bar)

        if zero_sum:
            player_count = len(b_bar)
            S = np.column_stack(
                (np.eye(player_count - 1), -np.ones(player_count - 1)),
            )
            b = b @ S.T @ np.linalg.inv(S @ S.T)

        parameters = np.linalg.lstsq(A, b)[0]

        if zero_sum:
            parameters = parameters @ S

        return LinearValueFunction(feature_extractor, parameters)

    def __call__(self, terminal_history: _H) -> Any:
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

    return np.average(mapped_values, 0, weights)
