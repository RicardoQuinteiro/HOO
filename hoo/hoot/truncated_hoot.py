"""
Module that implements truncated Hierarchical Optimistic Optimization
applied to Trees (t-HOOT)
"""
from typing import Optional

from hoo.hoot.hoot import HOOT
from hoo.hoot.truncated_hoot_node import tHOOTNode
from hoo.state_actions.hoo_state import HOOState


class tHOOT(HOOT):

    def __init__(
        self,
        search_depth: int,
        initial_state: HOOState,
        gamma: float = 0.99,
        v1: Optional[float] = None,
        ce: float = 1.,
    ):
        """
        Initializes the truncated HOOT algorithm

        Args:
            search_depth: maximum search depth of the tree, which corresponds
                to the number of maximum consecutive actions that the algorithm
                will simulate in a run
            initial_state: state at which one wants to make a decision
            ldhoo_max_depth: maximum depth of the truncated-HOO tree
            gamma: discount factor
            v1: constant used in HOO
            ce: exploration constant that gives more emphasis to exploring
                less appealing nodes the higher it is
        """
        self.search_depth = search_depth

        self.root = tHOOTNode(
            initial_state,
            gamma=gamma,
            v1=v1,
            ce=ce,
        )
