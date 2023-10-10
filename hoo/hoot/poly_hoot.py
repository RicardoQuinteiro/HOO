"""
Module that implements Polynomial Hierarchical Optimistic Optimization
applied to Trees (Poly-HOOT)

Paper: POLY-HOOT: Monte-Carlo Planning in Continuous Space MDPs with
Non-Asymptotic Analysis
https://arxiv.org/abs/2006.04672
"""
from hoo.hoot.hoot import HOOT
from hoo.hoot.poly_hoot_node import PolyHOOTNode
from hoo.poly_hoo import PolyHOOConstants
from hoo.state_actions.hoo_state import HOOState
from hoo.experiments.run_configs import PolyHOOTRunConfigs


class PolyHOOT(HOOT):

    @classmethod
    def from_configs(
        cls,
        configs: PolyHOOTRunConfigs,
        initial_state: HOOState,
    ):
        """
        Initializes Poly-HOOT from run configs and an initial state

        Args:
            configs: a set of configurations for a HOOT run
            initial_state: the initial state for the run
        Returns:
            An instance of Poly-HOOT initialized from the run configs
        """
        polyhoo_constants = PolyHOOConstants(
            alpha=configs.alpha,
            eta=configs.eta,
            xi=configs.xi,
        )

        root = PolyHOOTNode(
            initial_state,
            configs.hoo_max_depth,
            gamma=configs.gamma,
            v1=configs.v1,
            ce=configs.ce,
            polyhoo_constants=polyhoo_constants,
        )

        return cls(
            configs.search_depth,
            root,
        )
