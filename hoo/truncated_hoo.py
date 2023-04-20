"""
Module that implements Truncated Hierarchical Optimistic Optimization (t-HOO)
https://arxiv.org/abs/1001.4475
"""
import math
from typing import List, Optional

from hoo.hoo import HOO
from hoo.state_actions.hoo_state import HOOState


class tHOO(HOO):

    def __init__(
        self,
        state: HOOState,
        v1: Optional[float] = None,
        ce: float = 1.,
    ):
        """
        Initializes Truncated HOO algorithm

        Args:
            state: initial state
            v1: parameter of the algorithm as defined in the paper
            ce: exploration constant that gives more emphasis to exploring
                less appealing nodes the higher it its
        """
        super().__init__(state, v1=v1, ce=ce)

    def run(self, n: int) -> List[float]:
        """
        Runs n iterations of tHOO

        Args:
            n: number of iterations to run the algorithm
        Returns:
            A recommended action sampled from the best node
        """
        self.n = n

        return super().run(n)

    def update_U_B(self):
        """
        Updates the values of the U and B-values of the HOO tree's nodes

        The update is done only on the nodes that belong to the path followed
        in a certain iteration of the algorithm
        """

        # Updating U-values of the nodes in the path
        for node in self.path:
            node.U = (
                node.R / node.N
                + self.ce * math.sqrt((2.0 * math.log(self.n)) / node.N)
                + self.v1 * (self.rho**node.h)
            )
        node = self.path[-1]

        # Updating the B-values of the nodes in the path from bottom to top
        while not node.root():
            node.B = min(node.U, max([x.B for x in node.children]))
            node = node.parent

    def backpropagate(self, reward: float, t: int) -> None:
        """
        Updates visited nodes information and all nodes' U and B-values

        Updates the count and cumulative reward of the visited nodes and
        then does tree's U and B-values update

        Args:
            reward: reward to be backpropagated
            t: time-step of the algorithm
        """

        for node in self.path:
            node.N += 1
            node.R += reward

        self.update_U_B()
