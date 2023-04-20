"""
Module that implements Hierarchical Optimistic Optimization (HOO)
https://arxiv.org/abs/1001.4475
"""
import math
from typing import List, Optional

# import numpy as np
# import matplotlib.pyplot as plt

from hoo.hoo_node import HOONode
from hoo.state_actions.hoo_state import HOOState


class HOO:

    def __init__(
        self,
        state: HOOState,
        v1: Optional[float] = None,
        ce: float = 1.,
    ):
        """
        Initializes the HOO algorithm

        Args:
            state: initial state
            v1: parameter of the algorithm as defined in the paper
            ce: exploration constant that gives more emphasis to exploring
                less appealing nodes the higher it is
        """
        self.state = state
        self.root = HOONode(state.action_space)
        self.m = self.root.dimension

        if v1:
            self.v1 = v1
        else:
            self.v1 = 4 * self.m

        self.rho = 1.0 / (4**self.m)
        self.ce = ce

        self.path = []

    def run(self, n: int) -> List[float]:
        """
        Runs n iterations of HOO

        Args:
            n: number of iterations to run the algorithm
        Returns:
            A recommended action sampled from the best node
        """
        for t in range(n):
            selected_node = self.generate_path()

            action = selected_node.sample()
            reward = self.state.simulate(action).reward

            self.backpropagate(reward, t+1)

        best_node = self.choose_best_node(self.root)
        return best_node.sample()

    def generate_path(self) -> None:
        """
        Generates a path of nodes in the HOO tree

        Selects and keeps a trace on the nodes by following those with the
        highest B-values until a leaf is reached. Then, generates its children
        nodes

        Returns:
            Leaf node that is at the end of the followed path
        """

        node = self.root
        self.path = [node]

        while not node.leaf():
            node = node.choose_child()
            self.path += [node]

        node.generate_children()

        return node

    def backpropagate(self, reward: float, t: int) -> None:
        """
        Updates visited nodes information and all nodes' U and B-values

        Updates the count and cumulative reward of the visited nodes and
        then does tree's U and B-values update

        Args:
            reward: reward to be backpropagated
            t: time-step of the algorithm
        """

        for node in self.path:
            node.N += 1
            node.R += reward

        self.update_U_B(self.root, t)

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
                + self.ce * math.sqrt((2.0 * math.log(t)) / node.N)
                + self.v1 * (self.rho**node.h)
            )

        for child in node.children:
            self.update_U_B(child, t)

        if node.leaf():
            self.update_B(node)

    def update_B(self, node: HOONode) -> None:
        """
        Updates the values of the B-values of the HOO tree's nodes

        The update is done from bottom to top, starting in the input node.
        The node's ready attribute is set to False whenever it is visited
        for the first time as most of the nodes will be visited more than
        once. This way the algorithm will only update the B-value once for
        each node, as it is supposed to.

        Args:
            node: node where the recursion of the B-values updates begins
        """
        if node.ready:
            node.ready = False

            if node.leaf():
                node.B = node.U
            else:
                node.B = min(node.U, max([x.B for x in node.children]))

            if not node.root():
                self.update_B(node.parent)

    def choose_best_node(self, node: HOONode) -> HOONode:
        """
        Recursively finds the node with the highest average reward

        Args:
            node: node to start the recursion from top to bottom
        Returns:
            The node with the current highest average reward
        """
        if node.leaf():
            return node
        else:
            current_max = node.R / node.N
            current_child = node

            for child in node.children:
                child_best = self.choose_best_node(child)

                if child_best.average_reward() > current_max:
                    current_child = child_best
                    current_max = child_best.average_reward()

            if current_max >= node.average_reward():
                return current_child
            else:
                return node

    def choose_best_action(self):
        """
        Returns an action sampled from the best node

        Returns:
            An action sampled from the node with the current highest
                average reward
        """
        best_node = self.choose_best_node(self.root)
        return best_node.sample()

    """
    def depth(self, node):
        if node.leaf():
            return 1
        else:
            return 1 + max(
                self.depth(node.children[0]), self.depth(node.children[1])
            )

    def compute_for_plot(self, node, depth=False):
        if node.leaf():
            if depth:
                return [[node.center], [node.h], [node.N]]
            else:
                return [[], []]
        else:
            x = []
            y = []
            s = []
            for child in node.children:
                result = self.compute_for_plot(child, depth=depth)
                x += result[0]
                y += result[1]
                if depth:
                    s += result[2]
        if depth:
            return [x + [node.center], y + [node.h], s + [1]]
        return [x + [node.center], y + [node.h]]

    def plot(self, f=False, show=True):
        if self.state.dimension() != 1:
            print("Plot not available for action space dimension != 1.")
            return

        depth = self.hoo_max_depth < float("inf")

        data = self.compute_for_plot(self.root, depth=depth)
        x = np.array(data[0])
        y = np.array(data[1])
        fx = np.linspace(0, 1, 200)

        plt.figure()

        if f:
            fy = [ftest(xx) for xx in fx]

            fig, ax1 = plt.subplots()
            ax1.plot(fx, fy, c="firebrick")
            ax1.set_xlabel("x")
            ax1.set_ylabel("reward")
            ax1.tick_params("y", colors="firebrick")

            ax2 = ax1.twinx()
            if depth:
                s = np.array(data[2])
                ax2.scatter(x, y, s=s, c="royalblue")
            else:
                ax2.scatter(x, y, s=10, c="royalblue")
            ax2.set_ylabel("depth")
            ax2.set_yticks([i for i in range(max(y) + 1)])
            ax2.tick_params("y", colors="royalblue")
            if show:
                plt.show()
            return

        if depth:
            s = np.array(data[2])
            plt.scatter(x, y, s=s, c="royalblue")
        else:
            plt.scatter(x, y, s=10, c="royalblue")

        if show:
            plt.show()"""
