"""
Module that implements Polynomial Hierarchical Optimistic Optimization
applied to Trees (LD-HOOT)

Limited depth bandit-based strategy for Monte Carlo planning in continuous
action spaces
https://arxiv.org/abs/2106.15594
"""
from hoo.hoot.hoot import HOOT
from hoo.hoot.ld_hoot_node import LDHOOTNode
from hoo.state_actions.hoo_state import HOOState
from hoo.experiments.run_configs import LDHOOTRunConfigs


class LDHOOT(HOOT):

    @classmethod
    def from_configs(cls, configs: LDHOOTRunConfigs, initial_state: HOOState):
        """
        Initializes LD-HOOT from run configs and an initial state

        Args:
            configs: a set of configurations for a HOOT run
            initial_state: the initial state for the run
        Returns:
            An instance of LD-HOOT initialized from the run configs
        """
        root = LDHOOTNode(
            initial_state,
            configs.hoo_max_depth,
            gamma=configs.gamma,
            v1=configs.v1,
            ce=configs.ce,
        )

        return cls(
            configs.search_depth,
            root,
        )
