"""Module that implements a HOO node"""
from __future__ import annotations

import math
from typing import List, Optional, Union

import numpy as np
import numpy.random as rnd

from hoo.state_actions.action_space import HOOActionSpace


class HOONode:
    """
    HOONode is a node class that is compatible with the following HOO variants
    implemented in this library: HOO, LD-HOO, truncated-HOO and Poly-HOO
    """

    def __init__(
        self,
        action_space: HOOActionSpace,
        max_depth: Union[int, float] = float("inf"),
        depth: int = 0,
        parent: Optional[HOONode] = None,
    ) -> None:
        """
        Initializes a HOONode

        Args:
            action_space: the action space where the search will be done
            max_depth: maximum depth of the tree search (used in LD-HOO and
                Poly-HOO)
            depth: depth of the node in the HOO tree
            parent: node that is above in the HOO tree
        """
        self.h = depth
        self.action_space = action_space

        self.parent = parent
        self.children = []

        self.R = 0
        self.N = 0
        self.U = math.inf
        self.B = math.inf
        self.max_depth = max_depth

        self.ready = False

        self.split_dimension = rnd.choice(np.arange(self.dimension))

    @property
    def dimension(self) -> int:
        return self.action_space.dim

    @property
    def low(self) -> List[float]:
        return self.action_space.low

    @property
    def high(self) -> List[float]:
        return self.action_space.high

    @property
    def center(self) -> List[float]:
        return self.action_space.center

    def is_max_depth(self) -> bool:
        return self.h == self.max_depth

    def leaf(self) -> bool:
        return len(self.children) == 0 or self.is_max_depth

    def root(self) -> bool:
        return self.h == 0

    def sample(self) -> List[float]:
        return self.action_space.sample()

    def generate_children(self) -> None:
        """
        Generates two new HOO nodes to the children, by splitting the action
        space into two halves
        """
        if not self.is_max_depth():
            lower_space, upper_space = self.action_space.split(
                self.split_dimension
            )

            self.children.append(
                HOONode(
                    lower_space,
                    max_depth=self.max_depth,
                    depth=self.h + 1,
                    parent=self,
                )
            )

            self.children.append(
                HOONode(
                    upper_space,
                    max_depth=self.max_depth,
                    depth=self.h + 1,
                    parent=self,
                )
            )

    def choose_child(self) -> HOONode:
        """
        Randomly chooses from the children nodes that have the highest
        B-value

        Returns:
            The selected children node
        """
        B_max = -math.inf
        best_children = []

        for child in self.children:

            if child.B > B_max:
                B_max = child.B
                best_children = [child]

            elif child.B == B_max:
                best_children.append(child)

        return rnd.choice(best_children)

    def average_reward(self) -> float:
        """
        Calculates the average reward of the node

        Return:
            Average reward
        """
        if self.N != 0:
            return self.R / self.N
        else:
            return -math.inf
