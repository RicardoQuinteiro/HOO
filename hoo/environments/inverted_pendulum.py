"""
Module that implements a modified version of Open AI gym's Inverted
Pendulum environment
"""
from copy import deepcopy

from gym.envs.classic_control import PendulumEnv

from hoo.environments.environment import Environment, StepOutput
from hoo.state_actions.action_space import HOOActionSpace


class InvertedPendulum(Environment, PendulumEnv):
    def __init__(self):
        PendulumEnv.__init__()

    def step(self, action):

        previous_state = deepcopy(self)
        _, reward, done, _, _ = super().step(action)

        return StepOutput(
            previous_state=previous_state, reward=reward, done=done
        )

    @property
    def hoo_action_space(self):
        return HOOActionSpace([(self.max_torque, self.max_torque)])
