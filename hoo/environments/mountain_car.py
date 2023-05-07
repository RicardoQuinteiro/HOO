"""
Module that implements a modified version of Open AI gym's Continuous
Mountain Car environment
"""
from copy import deepcopy
from typing import Optional

from gym.envs.classic_control import Continuous_MountainCarEnv

from hoo.environments.environment import Environment, StepOutput
from hoo.state_actions.action_space import HOOActionSpace


class MountainCar(Continuous_MountainCarEnv, Environment):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.reset(seed=seed)

    def step(self, action, clip_reward: bool = True):

        previous_state = deepcopy(self)
        _, reward, done, _, _ = super().step(action)

        if clip_reward:
            reward = (reward + 1) / 101

        return StepOutput(
            previous_state=previous_state, reward=reward, done=done
        )

    @property
    def hoo_action_space(self):
        return HOOActionSpace([(self.min_action, self.max_action)])
