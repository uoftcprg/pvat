""":mod:`pvat.tests.test_estimators` implements unit tests for the value
estimators in PVAT.
"""

from collections.abc import Iterator
from enum import IntEnum
from functools import partial
from unittest import TestCase, main

import numpy as np

from pvat.estimators import ImportanceSampling


class RockPaperScissorsTestCase(TestCase):
    class Hand(IntEnum):
        ROCK = 0
        PAPER = 1
        SCISSORS = 2

    def get_terminal_value(
            self,
            terminal_history: tuple[Hand | None, Hand | None],
    ) -> tuple[float, float]:
        hero_hand, villain_hand = terminal_history

        assert hero_hand is not None and villain_hand is not None

        if (hero_hand + 1) % 3 == villain_hand:
            value = 0.0, 1.0
        elif (hero_hand - 1) % 3 == villain_hand:
            value = 1.0, 0.0
        else:
            value = 0.5, 0.5

        return value

    def get_mapped_terminal_histories(
            self,
            terminal_history: tuple[Hand | None, Hand | None],
    ) -> Iterator[tuple[Hand | None, Hand | None]]:
        _, villain_hand = terminal_history

        for hand in self.Hand:
            yield hand, villain_hand

    def get_terminal_probability(
            self,
            hero_strategy: tuple[float, float, float],
            terminal_history: tuple[Hand | None, Hand | None],
    ) -> float:
        hero_hand, _ = terminal_history

        if hero_hand is None:
            probability = 1.0
        else:
            probability = hero_strategy[hero_hand]

        return probability

    def test_importance_sampling(self) -> None:
        importance_sampling = ImportanceSampling(
            self.get_terminal_value,
            self.get_mapped_terminal_histories,
            partial(self.get_terminal_probability, (1.0, 0.0, 0.0)),
        )

        for hand in self.Hand:
            np.testing.assert_allclose(
                importance_sampling((hand, self.Hand.ROCK)),
                (0.5, 0.5),
            )
            np.testing.assert_allclose(
                importance_sampling((hand, self.Hand.PAPER)),
                (0.0, 1.0),
            )
            np.testing.assert_allclose(
                importance_sampling((hand, self.Hand.SCISSORS)),
                (1.0, 0.0),
            )

        importance_sampling = ImportanceSampling(
            self.get_terminal_value,
            self.get_mapped_terminal_histories,
            partial(self.get_terminal_probability, (0.8, 0.1, 0.1)),
        )

        for hand in self.Hand:
            np.testing.assert_allclose(
                importance_sampling((hand, self.Hand.ROCK)),
                (0.5, 0.5),
            )
            np.testing.assert_allclose(
                importance_sampling((hand, self.Hand.PAPER)),
                (0.15, 0.85),
            )
            np.testing.assert_allclose(
                importance_sampling((hand, self.Hand.SCISSORS)),
                (0.85, 0.15),
            )

        importance_sampling = ImportanceSampling(
            self.get_terminal_value,
            self.get_mapped_terminal_histories,
            partial(self.get_terminal_probability, (1 / 3,) * 3),
        )

        for hand in self.Hand:
            np.testing.assert_allclose(
                importance_sampling((hand, self.Hand.ROCK)),
                (0.5, 0.5),
            )
            np.testing.assert_allclose(
                importance_sampling((hand, self.Hand.PAPER)),
                (0.5, 0.5),
            )
            np.testing.assert_allclose(
                importance_sampling((hand, self.Hand.SCISSORS)),
                (0.5, 0.5),
            )

    def test_divat(self) -> None:
        pass

    def test_mivat(self) -> None:
        pass

    def test_aivat(self) -> None:
        pass


class LeducHoldemTestCase(TestCase):
    def test_importance_sampling(self) -> None:
        pass

    def test_divat(self) -> None:
        pass

    def test_mivat(self) -> None:
        pass

    def test_aivat(self) -> None:
        pass


if __name__ == '__main__':
    main()  # pragma: no cover
