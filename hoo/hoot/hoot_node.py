"""Module that implements a HOOT Node"""
from __future__ import annotations

from typing import List, Optional, Tuple

from hoo.hoo import HOO
from hoo.state_actions.hoo_state import HOOState
from hoo.state_actions.hoo_state import SimulateOutput


class HOOTNode:

    def __init__(
        self,
        state: HOOState,
        parent: Optional[HOOTNode] = None,
        gamma: float = 0.99,
        depth: int = 0,
        v1: Optional[float] = None,
        ce: float = 1.,
    ) -> None:
        """
        Initializes and instance of a HOOTNode

        Args:
            state: a state of the simulation
            parent: node above in the HOOT tree
            gamma: discount factor
            depth: depth of the node in the HOOT tree
            v1: constant used in HOO
            ce: exploration constant that gives more emphasis to exploring
                less appealing nodes the higher it is
        """
        self.state = state
        self.parent = parent
        self.depth = depth
        self.gamma = gamma

        self.v1 = v1
        self.ce = ce

        self.hoo = HOO(state, v1=v1, ce=ce)

        self.children = {}

    def select_action(
        self,
        sample: bool = True,
        clip_reward: bool = True,
    ) -> Tuple[HOOTNode, SimulateOutput]:
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

        simulation_output = self.state.simulate(action, clip_reward)
        child_index = str(hoo_node.center)

        if self.children.get(child_index) is None:
            next_node = HOOTNode(
                simulation_output.next_state,
                parent=self,
                gamma=self.gamma,
                depth=self.depth + 1,
                v1=self.v1,
                ce=self.ce,
            )
            self.children[child_index] = next_node
        else:
            next_node = self.children[child_index]

        return next_node, simulation_output

    def backpropagate(
        self,
        rewards: List[float],
        t: int,
        clip_reward: bool = True,
    ) -> None:
        """
        Backpropagates the rewards through the HOOT tree

        Args:
            rewards: a list with the rewards obtained after one iteration of
                the HOOT tree search
            t: time-step
            max_reward: if given a max_reward, the rewards will be normalized
        """
        cumulative_reward = sum(
            [r*self.gamma**i for i, r in enumerate(rewards[self.depth:])]
        )

        if clip_reward:
            cumulative_reward = cumulative_reward / sum([
                self.gamma**i
                for i in range(len(rewards[self.depth:]))
            ])

        self.hoo.backpropagate(cumulative_reward, t)
        if not self.root():
            self.parent.backpropagate(rewards, t)

    def choose_best_action(self):
        """
        Returns an action sampled from the best node on this node's HOO tree

        Returns:
            An action sampled from the HOO node with the current highest
                average reward
        """
        return self.hoo.choose_best_action()

    def root(self) -> bool:
        return self.depth == 0

    def leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def max_reward(self) -> bool:
        return self.state.max_reward
