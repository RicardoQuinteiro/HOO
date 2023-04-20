"""Module that implements a HOO action space"""
from __future__ import annotations

from typing import List, Tuple

import numpy.random as rnd


class HOOActionSpace:

    space: List[Tuple(float, float)]

    def __init__(self, space: List[Tuple(float, float)]):
        self.space = space

    @property
    def dim(self) -> int:
        return len(self.space)

    @property
    def center(self) -> List[float]:
        [(a + b) / 2.0 for a, b in self.space]

    @property
    def low(self) -> List[float]:
        return [a for a, _ in self.space]

    @property
    def high(self) -> List[float]:
        return [b for _, b in self.space]

    def sample(self) -> List[float]:
        return [rnd.uniform(a, b) for a, b in self.space]

    def split(self, split_dimension) -> Tuple[HOOActionSpace]:
        """
        Splits the action space halfway two generate two partitions

        Returns:
            A tuple with the two halfway partitions of the space
        """
        boundary = (
            self.low[split_dimension] + self.high[split_dimension]
        ) / 2.0

        lower_space = HOOActionSpace(
            [
                t if i != split_dimension else (t[0], boundary)
                for i, t in enumerate(self.space)
            ]
        )

        upper_space = HOOActionSpace(
            [
                t if i != split_dimension else (boundary, t[1])
                for i, t in enumerate(self.space)
            ]
        )

        return lower_space, upper_space
