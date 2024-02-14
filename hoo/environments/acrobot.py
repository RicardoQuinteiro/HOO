"""
Module that implements a continuous version of OpenAI gym's Acrobot
environment

Partially copied from the original in:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py

Modified to have a continuous action space instead of the pre-built
discrete space
"""
from copy import deepcopy
from typing import Optional

import numpy as np
from gym import spaces
from gym.envs.classic_control import AcrobotEnv
from gym.envs.classic_control.acrobot import bound, rk4, wrap

from hoo.environments.environment import Environment, StepOutput
from hoo.state_actions.action_space import HOOActionSpace


class ContinuousAcrobot(AcrobotEnv, Environment):

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        clip_reward: bool = False,
    ):
        super().__init__(render_mode=render_mode)
        self.clip_reward = clip_reward

        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(1,), dtype=np.float32
        )

        self.reset(seed=seed)

    def step(self, a):

        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        torque = a[0]

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(
                -self.torque_noise_max, self.torque_noise_max
            )

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminated = self._terminal()
        reward = -1.0 if not terminated else 0.0

        if self.render_mode == "human":
            self.render()

        return StepOutput(
            reward=reward,
            done=terminated,
        )

    @property
    def hoo_action_space(self):
        return HOOActionSpace([(-1.0, 1.0)])
