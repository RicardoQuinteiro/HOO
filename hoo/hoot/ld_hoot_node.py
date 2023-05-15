"""Module that implements a LD-HOOT node"""
from __future__ import annotations

from typing import List, Optional

from hoo.state_actions.hoo_state import HOOState
from hoo.ld_hoo import LDHOO
from hoo.hoot.hoot_node import HOOTNode


class LDHOOTNode(HOOTNode):

    def __init__(
        self,
        state: HOOState,
        ldhoo_max_depth,
        parent: Optional[LDHOOTNode] = None,
        action: Optional[List] = None,
        gamma: float = 0.99,
        depth: int = 0,
        v1: Optional[float] = None,
        ce: float = 1.,
    ):
        """
        Initializes and instance of a LDHOOTNode

        Args:
            state: a state of the simulation
            ldhoo_max_depth: maximum depth of search of LD-HOO
            parent: node above in the LD-HOOT tree
            gamma: discount factor
            depth: depth of the node in the HOOT tree
            v1: constant used in LD-HOO
            ce: exploration constant that gives more emphasis to exploring
                less appealing nodes the higher it is
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
        self.hoo = LDHOO(
            state,
            ldhoo_max_depth,
            v1=v1,
            ce=ce,
        )
