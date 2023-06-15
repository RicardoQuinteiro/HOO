"""
Module that implements a modified version of Open AI gym's Lunar Lander
environment
"""
from copy import deepcopy
from typing import Optional

from gym.envs.box2d import LunarLander as LunarLanderEnv

from hoo.environments.environment import Environment, StepOutput
from hoo.state_actions.action_space import HOOActionSpace


class LunarLander(LunarLanderEnv, Environment):

    def __init__(self, seed: Optional[int] = None):
        super().__init__(continuous=True)
        self.reset(seed=seed)

    def step(self, action, clip_reward: bool = False):

        previous_state = deepcopy(self)
        _, reward, done, _, _ = super().step(action)

        return StepOutput(
            previous_state=previous_state, reward=reward, done=done
        )

    @property
    def hoo_action_space(self):
        return HOOActionSpace([(-1, 1), (-1, 1)])
