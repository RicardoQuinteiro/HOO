"""Module that implements a truncated-HOOT node"""
from __future__ import annotations

from typing import Optional

from hoo.state_actions.hoo_state import HOOState
from hoo.truncated_hoo import tHOO
from hoo.hoot.hoot_node import HOOTNode


class tHOOTNode(HOOTNode):

    def __init__(
        self,
        state: HOOState,
        parent: Optional[tHOOTNode] = None,
        gamma: float = 0.99,
        depth: int = 0,
        v1: Optional[float] = None,
        ce: float = 1.,
    ):
        """
        Initializes and instance of a tHOOTNode

        Args:
            state: a state of the simulation
            parent: node above in the truncated-HOOT tree
            gamma: discount factor
            depth: depth of the node in the HOOT tree
            v1: constant used in truncated-HOO
            ce: exploration constant that gives more emphasis to exploring
                less appealing nodes the higher it is
        """
        super().__init__(
            state,
            parent=parent,
            depth=depth,
            gamma=gamma,
        )

        self.vars = vars
        self.hoo = tHOO(
            state,
            v1=v1,
            ce=ce,
        )
