from abc import ABC, abstractmethod
from ..core.state import SwarmState


class Task(ABC):
    """
    Minimal task interface. Tasks consume SwarmState and return metrics/reward.
    """

    @abstractmethod
    def reset(self, state: SwarmState):
        ...

    @abstractmethod
    def compute(self, state: SwarmState) -> dict:
        ...
