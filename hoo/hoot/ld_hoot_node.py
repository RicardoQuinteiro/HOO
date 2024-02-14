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
        reward: Optional[float] = None,
        done: bool = False,
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
            reward=reward,
            done=done,
            parent=parent,
            action=action,
            gamma=gamma,
            depth=depth,
            v1=v1,
            ce=ce,
        )

        self.ldhoo_max_depth = ldhoo_max_depth

        self.vars = vars
        self.hoo = LDHOO(
            state,
            ldhoo_max_depth,
            v1=v1,
            ce=ce,
        )

    def select_action(
        self,
        sample: bool = True,
    ) -> LDHOOTNode:
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
            next_node = LDHOOTNode(
                simulation_output.next_state,
                self.ldhoo_max_depth,
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