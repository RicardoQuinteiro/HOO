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

        self.polyhoo_max_depth

        self.vars = vars
        self.hoo = PolyHOO(
            state,
            polyhoo_max_depth,
            v1=v1,
            ce=ce,
            polyhoo_constants=polyhoo_constants,
        )

    def select_action(
        self,
        sample: bool = True,
    ) -> PolyHOOTNode:
        """
        Selects an action using HOO

        Args:
            sample: if True the action that leads to the following state
                is randomly sampled from a HOO node. If False the selected
                action is the center of the action space
        Returns:
            A tuple with the node that follows from taking the selected action
                and an instance of SimulateOutput, which contains the next
                HOOState, the reward and a boolean that informs whether the
                next state is terminal or not.
        """
        hoo_node = self.hoo.generate_path()

        if sample:
            action = hoo_node.sample()
        else:
            action = hoo_node.center

        child_index = str(hoo_node.center)

        if child_index not in self.children:
            simulation_output = self.state.simulate(action)
            next_node = PolyHOOTNode(
                simulation_output.next_state,
                self.polyhoo_max_depth,
                reward=simulation_output.reward,
                done=simulation_output.done,
                parent=self,
                action=action,
                gamma=self.gamma,
                depth=self.depth + 1,
                v1=self.v1,
                ce=self.ce,
            )
            self.children[child_index] = next_node
        else:
            next_node = self.children[child_index]

        return next_node
