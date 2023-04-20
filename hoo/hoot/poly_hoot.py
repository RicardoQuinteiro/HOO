"""
Module that implements Polynomial Hierarchical Optimistic Optimization
applied to Trees (Poly-HOOT)

Paper: POLY-HOOT: Monte-Carlo Planning in Continuous Space MDPs with
Non-Asymptotic Analysis
https://arxiv.org/abs/2006.04672
"""
from typing import Optional

from hoo.hoot.hoot import HOOT
from hoo.hoot.poly_hoot_node import PolyHOOTNode
from hoo.poly_hoo import PolyHOOConstants
from hoo.state_actions.hoo_state import HOOState


class PolyHOOT(HOOT):

    def __init__(
        self,
        search_depth: int,
        initial_state: HOOState,
        polyhoo_max_depth: int,
        gamma: float = 0.99,
        v1: Optional[float] = None,
        ce: float = 1.,
        polyhoo_constants: PolyHOOConstants = PolyHOOConstants(),
    ):
        """
        Initializes the Poly-HOOT algorithm

        Args:
            search_depth: maximum search depth of the tree, which corresponds
                to the number of maximum consecutive actions that the algorithm
                will simulate in a run
            initial_state: state at which one wants to make a decision
            polyhoo_max_depth: maximum depth of the Poly-HOO tree
            gamma: discount factor
            v1: constant used in HOO
            ce: exploration constant that gives more emphasis to exploring
                less appealing nodes the higher it is
            polyhoo_constants: constants alpha, xi and eta used in Poly-HOO
        """
        self.search_depth = search_depth

        self.root = PolyHOOTNode(
            initial_state,
            polyhoo_max_depth,
            gamma=gamma,
            v1=v1,
            ce=ce,
            polyhoo_constants=polyhoo_constants,
        )
