"""
Module that implements an environment to test the optimization of a function
"""
import math
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt

from hoo.environments.environment import Environment, StepOutput
from hoo.state_actions.action_space import HOOActionSpace


def default_function(x):
    return (math.sin(13*x)*math.sin(27*x) + 1) / 2


class TestFunction(Environment):

    def __init__(
        self,
        function: Callable = default_function,
        domain: List[Tuple[float, float]] = [(0, 1)],
    ):

        self.function = function
        self.domain = domain

    def step(self, action):

        return StepOutput(
            reward=self.function(action[0]),
            done=True,
        )

    @property
    def hoo_action_space(self):
        return HOOActionSpace(self.domain)

    def plot(self):
        if len(self.domain) != 1:
            print("Cannot plot a function that is not 1-dimensional")
        else:
            a, b = self.domain[0]
            xx = [a + (b - a)*i/500 for i in range(501)]
            yy = [self.function(x) for x in xx]

            plt.plot(xx, yy)
            plt.show()
