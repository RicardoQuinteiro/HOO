"""Module that implements an Environment abstract class"""
from __future__ import annotations

from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from hoo.state_actions.action_space import HOOActionSpace


@dataclass
class StepOutput:

    reward: float
    done: bool


class Environment(ABC):

    @abstractmethod
    def step(self, action, clip_reward: bool) -> StepOutput:
        pass

    @property
    @abstractmethod
    def hoo_action_space(self) -> HOOActionSpace:
        pass

    @abstractmethod
    def get_state(self) -> List:
        pass
