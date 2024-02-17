"""
Module that implements a continuous version of OpenAI gym's CartPole
environment

Partially copied from the original in:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

Modified to have a continuous action space instead of the pre-built
discrete space
"""
import math
from copy import deepcopy
from typing import Optional

import numpy as np
from gym import spaces
from gym.envs.classic_control import CartPoleEnv

from hoo.environments.environment import Environment, StepOutput
from hoo.state_actions.action_space import HOOActionSpace


class ContinuousCartPole(CartPoleEnv, Environment):

    def __init__(
        self,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
        tau: float = 0.02,
        force_mag: float = 10.0,
        seed: Optional[int] = None,
        clip_reward: bool = False,
    ):
        super().__init__()
        self.clip_reward = clip_reward

        self.force_mag = force_mag
        self.action_space = spaces.Box(
            -self.force_mag, self.force_mag, shape=(1,), dtype=np.float32
        )

        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masspole + self.masscart
        self.length = length
        self.polemass_length = self.masspole * self.length
        self.tau = tau

        self.reset(seed=seed)

    def step(self, action, clip_reward: bool = False):

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            action[0] + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = (
            temp - self.polemass_length * thetaacc * costheta / self.total_mass
        )

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot))

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        else:
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        return StepOutput(
            reward=reward,
            done=terminated,
        )

    @property
    def hoo_action_space(self):
        return HOOActionSpace([(-self.force_mag, self.force_mag)])
    
    def get_state(self):
        return [float(s) for s in self.state]


class IGContinuousCartPole(ContinuousCartPole):

    def __init__(self, seed: Optional[int] = None, clip_reward=False):
        super().__init__(
            gravity=50.,
            masscart=1.0,
            masspole=0.5,
            length=2.,
            tau=0.02,
            force_mag=10.0,
            seed=seed,
            clip_reward=clip_reward,
        )
