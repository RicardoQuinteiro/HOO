"""
Module that implements Hierarchical Optimistic Optimization
applied to Trees (HOOT)
"""
from typing import List, Optional

from hoo.hoot.hoot_node import HOOTNode
from hoo.state_actions.hoo_state import HOOState
from hoo.experiments.run_configs import HOOTRunConfigs


class HOOT:

    def __init__(
        self,
        search_depth: int,
        initial_state: HOOState,
        gamma: float = 0.99,
        v1: Optional[float] = None,
        ce: float = 1.,
        clip_reward: bool = False,
    ):
        """
        Initializes the HOOT algorithm

        Args:
            search_depth: maximum search depth of the tree, which corresponds
                to the number of maximum consecutive actions that the algorithm
                will simulate in a run
            initial_state: state at which one wants to make a decision
            gamma: discount factor
            v1: constant used in HOO
            ce: exploration constant that gives more emphasis to exploring
                less appealing nodes the higher it is
        """
        self.search_depth = search_depth
        self.clip_reward = clip_reward

        self.root = HOOTNode(
            initial_state,
            gamma=gamma,
            v1=v1,
            ce=ce,
        )

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
        return cls(
            configs.search_depth,
            initial_state,
            gamma=configs.gamma,
            v1=configs.v1,
            ce=configs.ce,
        )

    def run(self, n: int) -> List[float]:
        """
        Runs n iterations of HOOT

        Args:
            n: number of iterations to run the algorithm
        Returns:
            A recommended action sampled from the best node
        """
        for t in range(1, n + 1):
            last_node, rewards = self.search()
            self.backpropagate(last_node, rewards, t)

        return self.root.choose_best_action()

    def search(self):
        """
        Performs a search in the HOOT tree

        This function runs HOO to select the best action at a certain decision
        node of the tree and does so until it reaches the maximum search depth.
        It collects the rewards of doing the selected actions.

        Returns:
            A tuple with the final node of the search as well as a list of the
            collected rewards.
        """
        node = self.root
        rewards = []

        for _ in range(self.search_depth):
            node, simulation_output = node.select_action(
                clip_reward=self.clip_reward,
            )
            rewards.append(simulation_output.reward)

            if simulation_output.done:
                rewards += [
                    rewards[-1]
                    for _ in range(self.search_depth - len(rewards))
                ]
                break
        rewards.append(0)
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
        last_node.backpropagate(rewards, t, clip_reward=self.clip_reward)
