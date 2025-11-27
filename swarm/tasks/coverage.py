from ..core.metrics import coverage_extent, collision_count
from .base import Task
from ..core.state import SwarmState


class CoverageTask(Task):
    def __init__(self, target_area: float | None = None, collision_penalty: float = 0.0):
        self.target_area = target_area
        self.collision_penalty = collision_penalty
        self._initial_area = 0.0

    def reset(self, state: SwarmState):
        self._initial_area = coverage_extent(state)

    def compute(self, state: SwarmState) -> dict:
        area = coverage_extent(state)
        collisions = collision_count(state)
        reward = area - self._initial_area
        if self.target_area:
            reward = (area / self.target_area)
        reward -= collisions * self.collision_penalty
        return {
            "coverage_area": area,
            "collisions": collisions,
            "reward": reward,
        }
