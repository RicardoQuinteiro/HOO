"""Module that implements a Poly-HOOT node"""
from __future__ import annotations

from typing import List, Optional

from hoo.state_actions.hoo_state import HOOState
from hoo.poly_hoo import PolyHOO, PolyHOOConstants
from hoo.hoot.hoot_node import HOOTNode


class PolyHOOTNode(HOOTNode):

    def __init__(
        self,
        state: HOOState,
        polyhoo_max_depth,
        parent: Optional[PolyHOOTNode] = None,
        action: Optional[List] = None,
        gamma: float = 0.99,
        depth: int = 0,
        v1: Optional[float] = None,
        ce: float = 1.,
        polyhoo_constants: PolyHOOConstants = PolyHOOConstants(),
    ):
        """
        Initializes and instance of a PolyHOOTNode

        Args:
            state: a state of the simulation
            polyhoo_max_depth: maximum depth of search of Poly-HOO
            parent: node above in the Poly-HOOT tree
            gamma: discount factor
            depth: depth of the node in the HOOT tree
            v1: constant used in Poly-HOO
            ce: exploration constant that gives more emphasis to exploring
                less appealing nodes the higher it is
            polyhoo_constans: constants alpha, xi and eta used in Poly-HOO
        """
        super().__init__(
            state,
            parent=parent,
            action=action,
            gamma=gamma,
            depth=depth,
            v1=v1,
            ce=ce,
        )

        self.vars = vars
        self.hoo = PolyHOO(
            state,
            polyhoo_max_depth,
            v1=v1,
            ce=ce,
            polyhoo_constants=polyhoo_constants,
        )
