"""
Module that implements Polynomial Hierarchical Optimistic Optimization
applied to Trees (LD-HOOT)

Limited depth bandit-based strategy for Monte Carlo planning in continuous
action spaces
https://arxiv.org/abs/2106.15594
"""
from typing import Optional

from hoo.hoot.hoot import HOOT
from hoo.hoot.ld_hoot_node import LDHOOTNode
from hoo.state_actions.hoo_state import HOOState


class LDHOOT(HOOT):

    def __init__(
        self,
        search_depth: int,
        initial_state: HOOState,
        ldhoo_max_depth: int,
        gamma: float = 0.99,
        v1: Optional[float] = None,
        ce: float = 1.,
    ):
        """
        Initializes the LD-HOOT algorithm

        Args:
            search_depth: maximum search depth of the tree, which corresponds
                to the number of maximum consecutive actions that the algorithm
                will simulate in a run
            initial_state: state at which one wants to make a decision
            ldhoo_max_depth: maximum depth of the LD-HOO tree
            gamma: discount factor
            v1: constant used in HOO
            ce: exploration constant that gives more emphasis to exploring
                less appealing nodes the higher it is
        """
        self.search_depth = search_depth

        self.root = LDHOOTNode(
            initial_state,
            ldhoo_max_depth,
            gamma=gamma,
            v1=v1,
            ce=ce,
        )
