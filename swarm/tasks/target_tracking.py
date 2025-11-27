import numpy as np
from .base import Task
from ..core.state import SwarmState


class TargetTrackingTask(Task):
    def __init__(self, target_path, tolerance: float = 0.5):
        """
        target_path: callable time -> np.ndarray (position)
        """
        self.target_path = target_path
        self.tolerance = tolerance
        self._t = 0.0

    def reset(self, state: SwarmState):
        self._t = 0.0

    def compute(self, state: SwarmState) -> dict:
        self._t = state.t
        target = self.target_path(self._t)
        positions = np.array([a.pos for a in state.agents.values()])
        if len(positions) == 0:
            mean_dist = float("inf")
        else:
            mean_dist = float(np.linalg.norm(positions - target, axis=1).mean())
        success = mean_dist <= self.tolerance
        reward = -mean_dist
        return {
            "target": target,
            "mean_dist": mean_dist,
            "success": success,
            "reward": reward,
        }
