"""Module that implements an Environment abstract class"""
from __future__ import annotations

from abc import ABC, abstractmethod

from dataclasses import dataclass

from hoo.state_actions.action_space import HOOActionSpace


@dataclass
class StepOutput:

    previous_state: Environment
    reward: float
    done: bool


class Environment(ABC):

    @abstractmethod
    def step(self, action) -> StepOutput:
        pass

    @property
    @abstractmethod
    def hoo_action_space(self) -> HOOActionSpace:
        pass