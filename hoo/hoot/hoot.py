"""
Module that implements Hierarchical Optimistic Optimization
applied to Trees (HOOT)
"""
from typing import List

from hoo.hoot.hoot_node import HOOTNode
from hoo.state_actions.hoo_state import HOOState
from hoo.experiments.run_configs import HOOTRunConfigs


class HOOT:

    def __init__(self, search_depth: int, root: HOOTNode):
        """
        Initializes the HOOT algorithm
        """
        self.search_depth = search_depth
        self.root = root

    @classmethod
    def from_configs(cls, configs: HOOTRunConfigs, initial_state: HOOState):
        """
        Initializes HOOT from run configs and an initial state

        Args:
            configs: a set of configurations for a HOOT run
            initial_state: the initial state for the run
        Returns:
            An instance of HOOT initialized from the run configs
        """

        root = HOOTNode(
            initial_state,
            gamma=configs.gamma,
            v1=configs.v1,
            ce=configs.ce,
        )

        return cls(
            configs.search_depth,
            root,
        )

    def run(self, n: int, sample: bool = True) -> List[float]:
        """
        Runs n iterations of HOOT

        Args:
            n: number of iterations to run the algorithm
            sample: if True will sample an action from node's actions space,
                otherwise returns the center
        Returns:
            A recommended action sampled from the best node
        """
        for t in range(1, n + 1):
            last_node, rewards = self.search(sample=sample)
            self.backpropagate(last_node, rewards, t)

        return self.root.choose_best_action(sample=sample)

    def search(self, sample: bool = True):
        """
        Performs a search in the HOOT tree

        This function runs HOO to select the best action at a certain decision
        node of the tree and does so until it reaches the maximum search depth.
        It collects the rewards of doing the selected actions.

        Args:
            sample: if True will sample an action from node's actions space,
                otherwise returns the center
        Returns:
            A tuple with the final node of the search as well as a list of the
            collected rewards.
        """
        node = self.root
        rewards = []

        for _ in range(self.search_depth):
            node, simulation_output = node.select_action(
                sample=sample,
            )
            rewards.append(simulation_output.reward)

            if simulation_output.done:
                break

        rewards = rewards + [simulation_output.reward] * (self.search_depth
                                                          - len(rewards)) + [0.]
        return node, rewards

    def backpropagate(
        self,
        last_node: HOOTNode,
        rewards: List[float],
        t: int,
    ) -> None:
        """
        Backpropagation of the rewards through the tree

        Args:
            last_node: the final node of a HOOT search
            rewards: the list of the collected rewards
            t: current time-step
        """
        last_node.backpropagate(rewards, t)
