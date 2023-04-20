"""
Module that implements Polynomial Hierarchical Optimistic Optimization
(Poly-HOO)

Paper: POLY-HOOT: Monte-Carlo Planning in Continuous Space MDPs with
Non-Asymptotic Analysis
https://arxiv.org/abs/2006.04672
"""
from typing import Optional, Union

from pydantic import BaseModel

from hoo.hoo import HOO
from hoo.hoo_node import HOONode
from hoo.state_actions.hoo_state import HOOState


class PolyHOOConstants(BaseModel):
    alpha: float = 5.
    eta: float = 20.
    xi: float = 0.5


class PolyHOO(HOO):

    def __init__(
        self,
        state: HOOState,
        max_depth: Union[int, float],
        v1: Optional[float] = None,
        ce: float = 1.,
        polyhoo_constants: PolyHOOConstants = PolyHOOConstants(),
    ):
        """
        Initializes the Poly-HOO algorithm

        Args:
            state: initial state
            max_depth: max depth for the expansion of the algorithm's tree
            v1: parameter of the algorithm as defined in the paper
            ce: exploration constant that gives more emphasis to exploring
                less appealing nodes the higher it is
            polyhoo_constants: constants alpha, xi and eta used in Poly-HOO
        """
        super().__init__(state, v1=v1, ce=ce)

        self.root = HOONode(state.action_space, max_depth=max_depth)
        self.constants = polyhoo_constants

    def update_U_B(self, node: HOONode, t: int) -> None:
        """
        Updates the values of the U and B-values of the HOO tree's nodes

        The update is done from top to bottom, starting in the input node
        and updating its U-values. When it arrives to a leaf node, it will
        call update_B function, which will recursively update the B-values
        from bottom to top until the root node.

        Args:
            node: node where the recursion of the U and B-values updates begins
            t: time-step of the algorithm
        """
        node.ready = True

        if node.N > 0:
            node.U = (
                node.R / node.N
                + self.ce
                * t ** (self.constants.alpha / self.constants.xi)
                * node.N ** (self.constants.eta - 1)
                + self.v1 * (self.rho**node.h)
            )

        for child in node.children:
            self.update_U_B(child, t)

        if node.leaf():
            self.update_B(node)
