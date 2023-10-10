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

    def __init__(
            self,
            render_mode: Optional[str] = None,
            goal_velocity=0,
            seed: Optional[int] = None,
            clip_reward: bool = False
    ):
        super().__init__(render_mode=render_mode, goal_velocity=goal_velocity)
        self.reset(seed=seed)
        self.clip_reward = clip_reward

    def step(self, action):

        previous_state = deepcopy(self)
        _, reward, done, _, _ = super().step(action)

        if self.clip_reward:
            reward = (reward + 0.1) / 100.1

        return StepOutput(
            previous_state=previous_state, reward=reward, done=done
        )

    @property
    def hoo_action_space(self):
        return HOOActionSpace([(self.min_action, self.max_action)])
