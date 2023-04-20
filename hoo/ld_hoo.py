"""
Module that implements Limited Depth Hierarchical Optimistic Optimization
(LD-HOO) https://arxiv.org/abs/2106.15594
"""
from typing import Optional, Union

from hoo.hoo import HOO
from hoo.hoo_node import HOONode
from hoo.state_actions.hoo_state import HOOState


class LDHOO(HOO):

    def __init__(
        self,
        state: HOOState,
        max_depth:  Union[int, float],
        v1: Optional[float] = None,
        ce: float = 1.,
    ):
        """
        Initializes LD-HOO algorithm

        Args:
            state: initial state
            hoo_max_depth: sets the maximum depth H that the HOO tree can
                expand
            v1: parameter of the algorithm as defined in the paper
            ce: exploration constant that gives more emphasis to exploring
                less appealing nodes the higher it is
        """
        super().__init__(state, v1=v1, ce=ce)

        self.root = HOONode(state.action_space, max_depth=max_depth)
