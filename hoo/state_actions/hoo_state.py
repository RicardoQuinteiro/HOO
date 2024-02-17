"""Module that implements a HOO state"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from hoo.environments.environment import Environment


@dataclass
class SimulateOutput:

    next_state: HOOState
    reward: float
    done: bool


class HOOState:

    def __init__(self, env_state: Environment):
        """
        Initializes a HOOState instance

        Args:
            env_state: an instance of an Environment
        """
        self.env_state = env_state
        self.action_space = env_state.hoo_action_space

    def simulate(self, action):
        """
        Simulates an action in this state

        Args:
            action: an action to be simulated
        Returns:
            An instance of SimulateOutput which contains the next state,
                the reward of doing the input action and a boolean (done)
                that informs if the action leads to a terminal state
        """
        next_env_state = deepcopy(self)
        action_output = next_env_state.env_state.step(action)

        return SimulateOutput(
            next_state=next_env_state,
            reward=action_output.reward,
            done=action_output.done,
        )

    @property
    def dimension(self):
        return self.action_space.dim

    @property
    def max_reward(self):
        return self.env_state.max_reward
    
    def get_state(self):
        return self.env_state.get_state()
