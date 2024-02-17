"""
Module that implements a modified version of Open AI gym's Continuous
Mountain Car environment
"""
import math
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

        _, reward, done, _, _ = super().step(action)

        if self.clip_reward:
            reward = (reward + 0.1) / 100.1

        return StepOutput(reward=reward, done=done)

    @property
    def hoo_action_space(self):
        return HOOActionSpace([(self.min_action, self.max_action)])
    
    def get_state(self):
        return [float(s) for s in self.state]


class SmoothedMountainCar(MountainCar, Environment):

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

        super().step(action)

        position = self.state[0]
        velocity = self.state[1]

        done = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        if done:
            reward = 100.0
        else:
            reward = 1.66 + (position - self.goal_position) - math.pow(action[0], 2) * 0.1

        if self.clip_reward:
            reward = reward / 100.1

        return StepOutput(reward=reward, done=done)

    @property
    def hoo_action_space(self):
        return HOOActionSpace([(self.min_action, self.max_action)])