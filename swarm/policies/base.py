from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def build_observation(self, self_state, neighbor_msgs, visible_targets=None):
        ...

    @abstractmethod
    def act(self, obs, return_log_prob: bool = False):
        """Return action vector (delta_v)."""
        ...
