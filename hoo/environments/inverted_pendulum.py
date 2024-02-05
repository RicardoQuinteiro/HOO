"""
Module that implements a modified version of Open AI gym's Inverted
Pendulum environment
"""
from copy import deepcopy
from typing import Optional

from gym.envs.classic_control import PendulumEnv

from hoo.environments.environment import Environment, StepOutput
from hoo.state_actions.action_space import HOOActionSpace


class InvertedPendulum(PendulumEnv, Environment):
    def __init__(self, seed: Optional[int] = None, clip_reward=False):
        super().__init__()
        self.reset(seed=seed)
        self.clip_reward = clip_reward

    def step(self, action):

        _, reward, done, _, _ = super().step(action)

        if self.clip_reward:
            reward = (reward + 16.2736044) / 16.2736044

        return StepOutput(reward=reward, done=done)

    @property
    def hoo_action_space(self):
        return HOOActionSpace([(-self.max_torque, self.max_torque)])
