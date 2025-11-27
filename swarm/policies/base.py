from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def build_observation(self, self_state, neighbor_msgs):
        ...

    @abstractmethod
    def act(self, obs):
        """Return action vector, e.g. delta_v."""
        ...
